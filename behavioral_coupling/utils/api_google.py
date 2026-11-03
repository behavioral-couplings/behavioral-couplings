import os

from google import genai
from google.genai import types

from behavioral_coupling.env import load_env
from behavioral_coupling.utils.api import response_result

load_env()


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


class GoogleLLM:
    def __init__(self, model_name, api_key=None):
        self.model = model_name
        if _env_flag("GOOGLE_GENAI_USE_VERTEXAI"):
            project = os.getenv("GOOGLE_CLOUD_PROJECT")
            location = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
            if not project:
                raise ValueError("GOOGLE_CLOUD_PROJECT must be set when GOOGLE_GENAI_USE_VERTEXAI=true.")
            self.client = genai.Client(vertexai=True, project=project, location=location)
        else:
            api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY must be set when not using Vertex AI.")
            self.client = genai.Client(api_key=api_key)

    def generate_result(
        self,
        user_prompt="",
        system_prompt="",
        thinking_budget=128,
        max_tokens=1024,
        temperature=1.0,
        top_p=1,
    ):
        config_kwargs = {
            "system_instruction": system_prompt,
            "max_output_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if thinking_budget > 0:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                include_thoughts=True,
                thinking_budget=thinking_budget,
            )
        else:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                include_thoughts=False,
                thinking_budget=0,
            )
        response = self.client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        if not response.candidates or response.candidates[0].finish_reason != types.FinishReason.STOP:
            raise ValueError("Incomplete Google response")
        parts = response.candidates[0].content.parts or []
        text = "".join(part.text for part in parts if part.text and not part.thought)
        return response_result(text, response.model_version or self.model,
            response.usage_metadata.model_dump(mode="json") if response.usage_metadata else {}, "STOP")

    def generate(self, *args, **kwargs):
        return self.generate_result(*args, **kwargs)["text"]
