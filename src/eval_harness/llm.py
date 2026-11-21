"""Thin multi-provider LLM client with a response cache and transient-error backoff.

Wraps litellm so one interface reaches OpenAI, Anthropic, or local Ollama, and returns a
small provider-neutral ``LLMResponse`` (text, tool calls, tokens, cost, latency). Agents
depend on the ``LLM`` callable type, not on litellm, so tests and the CI smoke eval can
inject a scripted fake.

- **Cache** (diskcache, on disk): keyed by hash(model, temperature, messages, tools,
  provider options). A hit replays the original response, including the cost and latency
  measured when it really ran, flagged ``cached=True``. Re-runs are free and reproducible.
- **Backoff** (tenacity): retries 408/429/5xx, connection errors, and timeouts with
  exponential backoff + jitter, bounded by attempts and by the caller's deadline.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from tenacity import (
    RetryError,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential_jitter,
)

from eval_harness.settings import env, repo_root

# Ollama defaults to a 4096-token context and silently drops the start of longer
# conversations. Agent transcripts (schema + tool results + retries) can exceed that.
OLLAMA_NUM_CTX = int(env("OLLAMA_NUM_CTX", "8192"))

CACHE_VERSION = 1  # bump to invalidate every cached response
CACHE_DIR = Path(env("EVAL_HARNESS_CACHE_DIR", str(repo_root() / ".cache" / "llm")))
CACHE_DISABLED = env("EVAL_HARNESS_NO_CACHE", "") not in ("", "0", "false")

MAX_ATTEMPTS = 5
TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504}
TRANSIENT_NAMES = {
    "RateLimitError",
    "InternalServerError",
    "ServiceUnavailableError",
    "BadGatewayError",
    "APIConnectionError",
    "Timeout",
    "APITimeoutError",
}


@dataclass(frozen=True)
class ToolCallRequest:
    """A tool call the model asked for. ``args`` is None if the model sent malformed JSON."""

    id: str
    name: str
    args: dict[str, Any] | None
    raw_args: str = ""


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0  # as measured when the call really ran (replayed on cache hits)
    cached: bool = False


class LLM(Protocol):
    def __call__(
        self, messages: list[dict], tools: list[dict], timeout: float | None = None
    ) -> LLMResponse: ...


def litellm_model_id(model: str) -> str:
    """Normalize config model ids. ``ollama/x`` -> ``ollama_chat/x``: litellm's chat route
    is the one that supports native tool calling for Ollama."""
    if model.startswith("ollama/"):
        return "ollama_chat/" + model.removeprefix("ollama/")
    return model


def is_local(model: str) -> bool:
    return model.startswith(("ollama/", "ollama_chat/"))


def is_transient(exc: BaseException) -> bool:
    """Worth retrying: rate limits, 5xx, connection drops, timeouts. Not auth/bad requests."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status in TRANSIENT_STATUS:
        return True
    return type(exc).__name__ in TRANSIENT_NAMES or isinstance(exc, TimeoutError | ConnectionError)


def _parse_args(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def cache_key(model: str, temperature: float, messages: list, tools: list, options: dict) -> str:
    payload = {
        "v": CACHE_VERSION,
        "model": litellm_model_id(model),
        "temperature": temperature,
        "messages": messages,
        "tools": tools,
        "options": options,
    }
    blob = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def _to_cache(resp: LLMResponse) -> dict:
    d = asdict(resp)
    d.pop("cached")
    return d


def _from_cache(d: dict) -> LLMResponse:
    calls = [ToolCallRequest(**c) for c in d.pop("tool_calls")]
    return LLMResponse(tool_calls=calls, cached=True, **d)


class LiteLLMClient:
    """Calls a model through litellm. One instance per (model, temperature).

    ``completion_fn`` defaults to ``litellm.completion``; tests inject a fake.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        request_timeout_s: float = 300,
        use_cache: bool = not CACHE_DISABLED,
        cache_dir: Path | str = CACHE_DIR,
        completion_fn: Callable[..., Any] | None = None,
        max_attempts: int = MAX_ATTEMPTS,
    ):
        self.model = model
        self.temperature = temperature
        self.request_timeout_s = request_timeout_s
        self.max_attempts = max_attempts
        self.wait = wait_exponential_jitter(initial=1, max=30)  # tests swap in wait_none()
        self._completion_fn = completion_fn
        self._cache = None
        if use_cache:
            import diskcache

            self._cache = diskcache.Cache(str(cache_dir))

    @property
    def options(self) -> dict[str, Any]:
        return {"num_ctx": OLLAMA_NUM_CTX} if is_local(self.model) else {}

    def _completion(self, **kwargs: Any) -> Any:
        if self._completion_fn is None:
            import litellm  # lazy: slow import, not needed by scripted tests

            litellm.suppress_debug_info = True  # no "Give Feedback" banners in eval output
            self._completion_fn = litellm.completion
        return self._completion_fn(**kwargs)

    def __call__(
        self, messages: list[dict], tools: list[dict], timeout: float | None = None
    ) -> LLMResponse:
        key = cache_key(self.model, self.temperature, messages, tools, self.options)
        if self._cache is not None:
            hit = self._cache.get(key)
            if hit is not None:
                return _from_cache(dict(hit))

        budget = min(self.request_timeout_s, timeout) if timeout else self.request_timeout_s
        start = time.perf_counter()
        retrying = Retrying(
            retry=retry_if_exception(is_transient),
            stop=stop_after_attempt(self.max_attempts) | stop_after_delay(max(budget, 1)),
            wait=self.wait,
            reraise=True,
        )
        try:
            raw = retrying(
                self._completion,
                model=litellm_model_id(self.model),
                messages=messages,
                tools=tools or None,
                temperature=self.temperature,
                timeout=budget,
                **self.options,
            )
        except RetryError as exc:  # pragma: no cover (reraise=True re-raises the original)
            raise exc.last_attempt.exception() from exc
        latency_ms = int((time.perf_counter() - start) * 1000)

        resp = self._parse(raw, latency_ms)
        if self._cache is not None:
            self._cache.set(key, _to_cache(resp))
        return resp

    def _parse(self, raw: Any, latency_ms: int) -> LLMResponse:
        msg = raw.choices[0].message
        calls = []
        for i, c in enumerate(getattr(msg, "tool_calls", None) or []):
            args = c.function.arguments
            calls.append(
                ToolCallRequest(
                    id=getattr(c, "id", None) or f"call_{i}",
                    name=c.function.name,
                    args=_parse_args(args),
                    raw_args=args if isinstance(args, str) else json.dumps(args),
                )
            )
        usage = getattr(raw, "usage", None)
        return LLMResponse(
            content=msg.content,
            tool_calls=calls,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            cost_usd=self._cost(raw),
            latency_ms=latency_ms,
        )

    def _cost(self, raw: Any) -> float:
        if is_local(self.model):
            return 0.0
        try:
            import litellm

            return float(litellm.completion_cost(completion_response=raw) or 0.0)
        except Exception:  # unknown pricing for a model must not fail the run
            return 0.0

    def warm_up(self) -> float:
        """One tiny uncached call so a local model is loaded before timed runs start.
        Returns seconds taken. No-op for hosted models."""
        if not is_local(self.model):
            return 0.0
        start = time.perf_counter()
        self._completion(
            model=litellm_model_id(self.model),
            messages=[{"role": "user", "content": "Reply with OK."}],
            temperature=0,
            timeout=self.request_timeout_s,
            **self.options,
        )
        return time.perf_counter() - start
