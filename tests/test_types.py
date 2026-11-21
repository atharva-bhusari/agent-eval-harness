"""Contract tests for eval_harness.types: the foundation every other module builds on."""

import dataclasses

import pytest

from eval_harness.types import (
    FAILURE_CATEGORIES,
    Agent,
    AgentResult,
    Config,
    RunRecord,
    Task,
    make_run_id,
)


def _record(**overrides) -> RunRecord:
    base = dict(
        run_id="x",
        task_id="t1",
        config_id="baseline",
        model="openai/gpt-4o-mini",
        prompt_version="v1",
        success=True,
        failure_category=None,
        latency_ms=10,
        prompt_tokens=1,
        completion_tokens=1,
        cost_usd=0.0,
        n_tool_calls=0,
        n_retries=0,
        final_sql=None,
        ts="2026-11-15T00:00:00Z",
    )
    base.update(overrides)
    return RunRecord(**base)


def test_task_is_frozen():
    task = Task(id="t1", question="q", expected_result=[(1,)])
    with pytest.raises(dataclasses.FrozenInstanceError):
        task.question = "changed"  # type: ignore[misc]


def test_task_rejects_unknown_difficulty():
    with pytest.raises(ValueError):
        Task(id="t1", question="q", expected_result=[], difficulty="impossible")


def test_task_defaults():
    task = Task(id="t1", question="q", expected_result=[])
    assert task.difficulty == "medium"
    assert task.ordered is False
    assert task.expected_sql is None


def test_config_defaults_are_deterministic():
    cfg = Config(id="baseline", model="openai/gpt-4o-mini", prompt_version="v1")
    assert cfg.temperature == 0.0
    assert cfg.max_iters == 12


def test_failure_taxonomy_matches_spec():
    assert set(FAILURE_CATEGORIES) == {
        "sql_syntax_error",
        "schema_hallucination",
        "wrong_result",
        "empty_result",
        "timeout",
        "max_iterations",
        "wrong_tool",
        "agent_error",
        "other",
    }


def test_run_record_failure_category_invariants():
    _record()  # success with no category: ok
    _record(success=False, failure_category="timeout")  # failure with valid category: ok
    with pytest.raises(ValueError):
        _record(success=False, failure_category=None)
    with pytest.raises(ValueError):
        _record(success=False, failure_category="made_up")
    with pytest.raises(ValueError):
        _record(success=True, failure_category="timeout")


def test_run_id_is_deterministic_and_distinct():
    a = make_run_id("t1", "baseline", "v1")
    assert a == make_run_id("t1", "baseline", "v1")
    assert a != make_run_id("t1", "baseline", "v2")
    assert a != make_run_id("t2", "baseline", "v1")


def test_agent_protocol_is_structural():
    class Dummy:
        def run(self, question: str) -> AgentResult:
            return AgentResult(None, None, [], 0, 0, 0, 0.0, 0)

    assert isinstance(Dummy(), Agent)
