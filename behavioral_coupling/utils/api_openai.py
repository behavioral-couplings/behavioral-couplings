import os

from openai import OpenAI

from behavioral_coupling.env import load_env
from behavioral_coupling.utils.api import response_result

load_env()


class OpenAILLM:
    def __init__(self, model_name, base_url=None, api_key=None, api="chat", reasoning_effort=None, timeout=120, max_retries=2):
        self.model = model_name
        self.api = api
        self.reasoning_effort = reasoning_effort
        if api not in {"chat", "responses"}:
            raise ValueError("api must be chat or responses")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY must be set.")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=timeout, max_retries=max_retries)

    def _uses_max_completion_tokens(self):
        return self.model.startswith(("gpt-5", "gpt-6", "o1", "o3", "o4"))

    def generate_result(
        self,
        user_prompt="",
        system_prompt="",
        max_tokens=1024,
        temperature=0.0,
        top_p=1.0,
        **_,
    ):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return self.chat_generate_result(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def generate(self, *args, **kwargs):
        return self.generate_result(*args, **kwargs)["text"]

    def chat_generate(self, *args, **kwargs):
        return self.chat_generate_result(*args, **kwargs)["text"]

    def chat_generate_result(
        self,
        messages,
        max_tokens=1024,
        temperature=0.0,
        top_p=1.0,
    ):
        if self.api == "responses":
            kwargs = dict(model=self.model, input=messages, max_output_tokens=max_tokens, store=False)
            if self.reasoning_effort is not None:
                kwargs["reasoning"] = {"effort": self.reasoning_effort}
            response = self.client.responses.create(**kwargs)
            if response.status != "completed" or not response.output_text.strip():
                raise ValueError(f"Incomplete or empty OpenAI response: {response.status}")
            usage = getattr(response, "usage", None)
            return response_result(response.output_text, getattr(response, "model", self.model),
                                   usage.model_dump() if usage else {}, response.status)
        completion_kwargs = {
            "model": self.model,
            "messages": messages,
        }
        if self._uses_max_completion_tokens():
            completion_kwargs["max_completion_tokens"] = max_tokens
            if self.reasoning_effort is not None:
                completion_kwargs["reasoning_effort"] = self.reasoning_effort
        else:
            completion_kwargs["max_tokens"] = max_tokens
            completion_kwargs["temperature"] = temperature
            completion_kwargs["top_p"] = top_p
        response = self.client.chat.completions.create(
            **completion_kwargs,
        )
        choice = response.choices[0]
        if choice.finish_reason != "stop":
            raise ValueError(f"Incomplete OpenAI response: {choice.finish_reason}")
        return response_result(choice.message.content, response.model,
                               response.usage.model_dump() if response.usage else {}, choice.finish_reason)
