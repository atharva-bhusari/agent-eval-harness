"""LLM client: response cache, transient-error backoff, response parsing. No network."""

from __future__ import annotations

import json
from types import SimpleNamespace as NS

import pytest
from tenacity import wait_none

from eval_harness.llm import LiteLLMClient, cache_key, is_transient


def fake_raw(content="ok", tool_calls=None, prompt_tokens=10, completion_tokens=2):
    msg = NS(content=content, tool_calls=tool_calls)
    return NS(
        choices=[NS(message=msg)],
        usage=NS(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


class FakeCompletion:
    """Stands in for litellm.completion: replays outcomes, counts calls."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes) or [fake_raw()]
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        out = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(out, Exception):
            raise out
        return out


class HTTPError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def client(tmp_path, fake, model="openai/gpt-4o-mini", **kw) -> LiteLLMClient:
    c = LiteLLMClient(model, cache_dir=tmp_path / "cache", completion_fn=fake, **kw)
    c.wait = wait_none()
    c._cost = lambda raw: 0.002  # deterministic pricing for tests
    return c


MSGS = [{"role": "user", "content": "How many orders?"}]


# --------------------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------------------
def test_cache_hit_skips_provider_and_replays_cost_and_latency(tmp_path):
    fake = FakeCompletion(fake_raw("first"))
    c = client(tmp_path, fake)
    a = c(MSGS, [])
    b = c(MSGS, [])
    assert len(fake.calls) == 1
    assert (a.cached, b.cached) == (False, True)
    assert b.content == "first"
    assert b.cost_usd == a.cost_usd == 0.002  # nominal cost replayed
    assert b.latency_ms == a.latency_ms  # nominal latency replayed


def test_cache_survives_a_new_client_instance(tmp_path):
    fake = FakeCompletion()
    client(tmp_path, fake)(MSGS, [])
    assert client(tmp_path, fake)(MSGS, []).cached
    assert len(fake.calls) == 1


def test_cache_misses_when_anything_in_the_request_changes(tmp_path):
    fake = FakeCompletion()
    c = client(tmp_path, fake)
    c(MSGS, [])
    c([{"role": "user", "content": "How many customers?"}], [])  # different messages
    c(MSGS, [{"type": "function", "function": {"name": "x"}}])  # different tools
    client(tmp_path, fake, temperature=0.7)(MSGS, [])  # different temperature
    client(tmp_path, fake, model="anthropic/claude-haiku-5-5")(MSGS, [])  # different model
    assert len(fake.calls) == 5


def test_cache_can_be_disabled(tmp_path):
    fake = FakeCompletion()
    c = client(tmp_path, fake, use_cache=False)
    c(MSGS, [])
    assert not c(MSGS, []).cached
    assert len(fake.calls) == 2


def test_cache_round_trips_tool_calls(tmp_path):
    call = NS(id="c1", function=NS(name="run_sql", arguments='{"query": "SELECT 1"}'))
    fake = FakeCompletion(fake_raw(None, tool_calls=[call]))
    c = client(tmp_path, fake)
    c(MSGS, [])
    (tc,) = c(MSGS, []).tool_calls
    assert (tc.id, tc.name, tc.args) == ("c1", "run_sql", {"query": "SELECT 1"})


def test_cache_key_is_stable_and_order_insensitive_for_dict_keys():
    k1 = cache_key("m", 0.0, [{"role": "user", "content": "x"}], [], {})
    k2 = cache_key("m", 0.0, [{"content": "x", "role": "user"}], [], {})
    assert k1 == k2
    assert k1 != cache_key("m", 0.0, [{"role": "user", "content": "y"}], [], {})


def test_ollama_requests_carry_num_ctx(tmp_path):
    fake = FakeCompletion()
    client(tmp_path, fake, model="ollama/llama3.1:8b")(MSGS, [])
    assert fake.calls[0]["model"] == "ollama_chat/llama3.1:8b"
    assert fake.calls[0]["num_ctx"] >= 8192


# --------------------------------------------------------------------------------------
# Backoff
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_transient_errors_are_retried(tmp_path, status):
    fake = FakeCompletion(HTTPError(status), HTTPError(status), fake_raw("recovered"))
    resp = client(tmp_path, fake)(MSGS, [])
    assert resp.content == "recovered"
    assert len(fake.calls) == 3


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_non_transient_errors_fail_fast(tmp_path, status):
    fake = FakeCompletion(HTTPError(status), fake_raw())
    with pytest.raises(HTTPError):
        client(tmp_path, fake)(MSGS, [])
    assert len(fake.calls) == 1


def test_retries_are_bounded(tmp_path):
    fake = FakeCompletion(HTTPError(429))
    with pytest.raises(HTTPError):
        client(tmp_path, fake, max_attempts=3)(MSGS, [])
    assert len(fake.calls) == 3


def test_failures_are_not_cached(tmp_path):
    fake = FakeCompletion(HTTPError(401), fake_raw("ok"))
    c = client(tmp_path, fake)
    with pytest.raises(HTTPError):
        c(MSGS, [])
    assert c(MSGS, []).cached is False


def test_is_transient():
    assert is_transient(HTTPError(429))
    assert is_transient(TimeoutError())
    assert is_transient(type("RateLimitError", (Exception,), {})())
    assert not is_transient(HTTPError(401))
    assert not is_transient(ValueError("bad"))


# --------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------
def test_tool_call_args_may_be_dict_or_malformed(tmp_path):
    calls = [
        NS(id=None, function=NS(name="get_schema", arguments={"table": "orders"})),
        NS(id="x", function=NS(name="run_sql", arguments="{broken")),
    ]
    resp = client(tmp_path, FakeCompletion(fake_raw(None, tool_calls=calls)), use_cache=False)(
        MSGS, []
    )
    a, b = resp.tool_calls
    assert (a.id, a.args, json.loads(a.raw_args)) == (
        "call_0",
        {"table": "orders"},
        {"table": "orders"},
    )
    assert b.args is None and b.raw_args == "{broken"
    assert (resp.prompt_tokens, resp.completion_tokens) == (10, 2)


def test_local_models_cost_nothing(tmp_path):
    c = LiteLLMClient("ollama/llama3.1:8b", use_cache=False, completion_fn=FakeCompletion())
    assert c(MSGS, []).cost_usd == 0.0


# ---- per-config request params (Phase 7: Claude Haiku 5.5 needs both) -------------------
def test_temperature_none_is_omitted_from_the_request(tmp_path):
    fake = FakeCompletion()
    client(tmp_path, fake, temperature=None)(MSGS, [])
    assert "temperature" not in fake.calls[0]
    fake2 = FakeCompletion()
    client(tmp_path / "b", fake2, temperature=0.0)(MSGS, [])
    assert fake2.calls[0]["temperature"] == 0.0


def test_extra_params_are_sent_and_keyed(tmp_path):
    fake = FakeCompletion(fake_raw(), fake_raw())
    thinking_off = {"thinking": {"type": "disabled"}}
    client(tmp_path, fake, extra_params=thinking_off)(MSGS, [])
    assert fake.calls[0]["thinking"] == {"type": "disabled"}
    client(tmp_path, fake)(MSGS, [])  # same messages, different params: must not hit cache
    assert len(fake.calls) == 2


def test_cache_key_is_unchanged_when_no_extra_params(tmp_path):
    """Adding extra_params support must not invalidate existing cache entries."""
    c = client(tmp_path, FakeCompletion())
    assert c.options == {}
    assert cache_key(c.model, c.temperature, MSGS, [], c.options) == cache_key(
        "openai/gpt-4o-mini", 0.0, MSGS, [], {}
    )


def test_haiku_configs_omit_temperature_and_disable_thinking():
    from eval_harness.config import load_config
    from eval_harness.settings import repo_root

    for name in ("haiku_v1", "haiku_v2"):
        cfg = load_config(repo_root() / "configs" / f"{name}.yaml")
        assert cfg.model == "anthropic/claude-haiku-5-5"
        assert cfg.temperature is None
        assert cfg.llm_params == {"thinking": {"type": "disabled"}}
