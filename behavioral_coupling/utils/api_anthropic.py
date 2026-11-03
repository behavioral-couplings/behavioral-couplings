import json
import os
import urllib.error
import urllib.request

from behavioral_coupling.env import load_env
from behavioral_coupling.utils.api import APIError, response_result


class AnthropicLLM:
    def __init__(self, model_name, api_key=None, reasoning_effort="low", timeout=120, base_url=None):
        load_env()
        self.model = model_name
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY must be set")
        if reasoning_effort not in {"low", "medium", "high", "xhigh", "max"}:
            raise ValueError("Adaptive Anthropic models require low/medium/high/xhigh/max effort")
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout
        self.base_url = (base_url or os.getenv("ANTHROPIC_BASE_URL") or "https://api.anthropic.com").rstrip("/")

    def generate_result(self, user_prompt="", system_prompt="", max_tokens=4096, **_):
        payload = dict(model=self.model, max_tokens=max_tokens,
                       thinking={"type": "adaptive"}, output_config={"effort": self.reasoning_effort},
                       messages=[{"role": "user", "content": user_prompt}])
        if system_prompt:
            payload["system"] = system_prompt
        request = urllib.request.Request(self.base_url + "/v1/messages", data=json.dumps(payload).encode(),
            headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            raise APIError(f"Anthropic HTTP {exc.code}; check credentials/credits for 4xx errors", exc.code) from None
        if result["stop_reason"] != "end_turn":
            raise ValueError(f"Incomplete Anthropic response: {result['stop_reason']}")
        text = "".join(c["text"] for c in result["content"] if c["type"] == "text")
        return response_result(text, result["model"], result.get("usage"), result["stop_reason"])

    def generate(self, *args, **kwargs):
        return self.generate_result(*args, **kwargs)["text"]
