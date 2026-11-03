import time


class APIError(RuntimeError):
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


def is_terminal_error(exc):
    status = getattr(exc, "status_code", getattr(exc, "code", None))
    return isinstance(status, int) and 400 <= status < 500 and status not in {408, 409, 429}


def retry_call(call, max_retries=3):
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    for attempt in range(max_retries + 1):
        try:
            return call()
        except Exception as exc:
            if attempt == max_retries or is_terminal_error(exc):
                raise
            time.sleep(min(60, 2 ** attempt))


def response_result(text, model, usage=None, finish_reason=None):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Empty API response")
    return dict(text=text, model=model, usage=usage or {}, finish_reason=finish_reason)


def resolve_provider(model, provider="auto"):
    if provider != "auto":
        if provider not in {"openai", "google", "anthropic"}:
            raise ValueError(f"Unknown provider: {provider}")
        return provider
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    if model.startswith("gemini-"):
        return "google"
    raise ValueError("Unknown model provider; specify --judge-provider explicitly")


def resolve_effort(model, provider, effort=None):
    if effort is not None:
        return effort
    if provider == "anthropic" or model.startswith("gpt-6"):
        return "low"
    return "none" if provider == "openai" else None


def create_api(model_name, provider="auto", reasoning_effort=None, timeout=120):
    provider = resolve_provider(model_name, provider)
    effort = resolve_effort(model_name, provider, reasoning_effort)
    if provider == "openai":
        from behavioral_coupling.utils.api_openai import OpenAILLM
        return OpenAILLM(model_name, api="responses", reasoning_effort=effort, timeout=timeout, max_retries=0)
    if provider == "anthropic":
        from behavioral_coupling.utils.api_anthropic import AnthropicLLM
        return AnthropicLLM(model_name, reasoning_effort=effort, timeout=timeout)
    from behavioral_coupling.utils.api_google import GoogleLLM
    return GoogleLLM(model_name)


def add_judge_arguments(parser):
    parser.add_argument("--judge-model", "--judge_model_name", dest="judge_model_name", default="gemini-3.1-pro-preview")
    parser.add_argument("--judge-provider", "--judge_provider", dest="judge_provider", choices=["auto", "google", "openai", "anthropic"], default="auto")
    parser.add_argument("--reasoning-effort", "--reasoning_effort", dest="reasoning_effort", choices=["none", "low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--max-tokens", "--max_tokens", dest="max_tokens", type=int, default=4096)
    parser.add_argument("--thinking-budget", "--thinking_budget", dest="thinking_budget", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", "--top_p", dest="top_p", type=float, default=1.0)
    parser.add_argument("--max-retries", "--max_retries", dest="max_retries", type=int, default=3)
