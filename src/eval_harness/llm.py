"""Thin multi-provider LLM client.

Wraps litellm so one interface reaches OpenAI, Anthropic, or local Ollama, and returns a
small provider-neutral ``LLMResponse`` (text, tool calls, tokens, cost). Agents depend on
the ``LLM`` callable type, not on litellm, so tests and the CI smoke eval can inject a
scripted fake.

Phase 4 adds the on-disk response cache and 429/5xx backoff here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from eval_harness.settings import env

# Ollama defaults to a 4096-token context and silently drops the start of longer
# conversations. Agent transcripts (schema + tool results + retries) can exceed that.
OLLAMA_NUM_CTX = int(env("OLLAMA_NUM_CTX", "8192"))


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


class LLM(Protocol):
    def __call__(self, messages: list[dict], tools: list[dict]) -> LLMResponse: ...


def litellm_model_id(model: str) -> str:
    """Normalize config model ids. ``ollama/x`` -> ``ollama_chat/x``: litellm's chat route
    is the one that supports native tool calling for Ollama."""
    if model.startswith("ollama/"):
        return "ollama_chat/" + model.removeprefix("ollama/")
    return model


def is_local(model: str) -> bool:
    return model.startswith(("ollama/", "ollama_chat/"))


def _parse_args(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


class LiteLLMClient:
    """Calls a model through litellm. One instance per (model, temperature)."""

    def __init__(self, model: str, temperature: float = 0.0, request_timeout_s: float = 300):
        self.model = model
        self.temperature = temperature
        self.request_timeout_s = request_timeout_s

    def __call__(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        import litellm  # imported lazily: slow import, and not needed by scripted tests

        kwargs: dict[str, Any] = {}
        if is_local(self.model):
            kwargs["num_ctx"] = OLLAMA_NUM_CTX
        resp = litellm.completion(
            model=litellm_model_id(self.model),
            messages=messages,
            tools=tools or None,
            temperature=self.temperature,
            timeout=self.request_timeout_s,
            **kwargs,
        )
        msg = resp.choices[0].message
        calls = [
            ToolCallRequest(
                id=c.id or f"call_{i}",
                name=c.function.name,
                args=_parse_args(c.function.arguments),
                raw_args=c.function.arguments
                if isinstance(c.function.arguments, str)
                else json.dumps(c.function.arguments),
            )
            for i, c in enumerate(msg.tool_calls or [])
        ]
        usage = getattr(resp, "usage", None)
        cost = 0.0
        if not is_local(self.model):
            try:
                cost = float(litellm.completion_cost(completion_response=resp) or 0.0)
            except Exception:  # unknown pricing for a model must not fail the run
                cost = 0.0
        return LLMResponse(
            content=msg.content,
            tool_calls=calls,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            cost_usd=cost,
        )
