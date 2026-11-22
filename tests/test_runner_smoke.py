"""Runner end-to-end tests: real agent + runner + scorer + store, with fake LLMs (no network)."""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from eval_harness import db, golden, runner
from eval_harness.agent.text_to_sql import TextToSQLAgent
from eval_harness.config import load_config
from eval_harness.llm import LLMResponse, ToolCallRequest
from eval_harness.store import Store
from eval_harness.types import FAILURE_CATEGORIES, AgentResult, Config

needs_db = pytest.mark.skipif(
    not db.DEFAULT_DB_PATH.exists(), reason="run `make db` to build data/northwind.duckdb"
)


class OracleLLM:
    """Answers every golden question with its reference SQL, then finishes.

    Stateless (decides from the transcript), so it is safe under concurrency, which is
    what lets it double as a thread-safety test for the agent.
    """

    def __init__(self, tasks):
        self.sql_for = {t.question: t.expected_sql for t in tasks}

    def __call__(self, messages, tools, timeout=None):
        if messages[-1]["role"] == "tool":
            return LLMResponse(content="Done.", prompt_tokens=50, completion_tokens=3)
        question = next(m["content"] for m in messages if m["role"] == "user")
        sql = self.sql_for.get(question, "SELECT 1")
        args = {"query": sql}
        call = ToolCallRequest(id="c1", name="run_sql", args=args, raw_args=json.dumps(args))
        return LLMResponse(content=None, tool_calls=[call], prompt_tokens=100, completion_tokens=20)


def config(**kw) -> Config:
    return Config(id=kw.pop("id", "test_cfg"), model="fake/model", prompt_version="v1", **kw)


def run(cfg, tasks, store, agent, **kw):
    return asyncio.run(runner.run_eval(cfg, tasks, store, agent=agent, warm_up=False, **kw))


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "runs.sqlite") as s:
        yield s


# --------------------------------------------------------------------------------------
# Full golden set through the real agent
# --------------------------------------------------------------------------------------
@needs_db
def test_oracle_scores_every_task_concurrently_and_rerun_is_idempotent(store):
    tasks = golden.load_tasks()
    agent = TextToSQLAgent(OracleLLM(tasks))
    cfg = config(concurrency=5)

    first = run(cfg, tasks, store, agent)
    assert first.n == len(tasks)
    assert first.accuracy == 1.0, [r.task_id for r in first.records if not r.success]
    assert store.count("test_cfg") == len(tasks)

    second = run(cfg, tasks, store, agent)  # re-run: overwrite, never duplicate
    assert second.n == len(tasks)
    assert store.count("test_cfg") == len(tasks)


@needs_db
def test_records_carry_trace_answer_and_metadata(store):
    tasks = golden.load_tasks()[:1]
    run(config(), tasks, store, TextToSQLAgent(OracleLLM(tasks)))
    (row,) = store.runs()
    assert row["success"] == 1 and row["failure_category"] is None
    assert row["final_sql"] == tasks[0].expected_sql
    assert json.loads(row["answer_json"]) == [list(r) for r in tasks[0].expected_result]
    assert json.loads(row["transcript_json"])[1]["content"] == tasks[0].question
    assert row["difficulty"] == tasks[0].difficulty
    assert [c["name"] for c in store.tool_calls(row["run_id"])] == ["run_sql"]


@needs_db
def test_resume_skips_completed_tasks(store):
    tasks = golden.load_tasks()[:4]
    agent = TextToSQLAgent(OracleLLM(tasks))
    run(config(), tasks[:2], store, agent)
    summary = run(config(), tasks, store, agent, resume=True)
    assert summary.skipped == 2
    assert {r.task_id for r in summary.records} == {t.id for t in tasks[2:]}


@needs_db
def test_wrong_answers_get_a_valid_failure_category(store):
    tasks = golden.load_tasks()[:3]

    def wrong(messages, tools, timeout=None):
        if messages[-1]["role"] == "tool":
            return LLMResponse(content="Done.")
        args = {"query": "SELECT 12345"}
        return LLMResponse(None, [ToolCallRequest("c", "run_sql", args, json.dumps(args))])

    summary = run(config(), tasks, store, TextToSQLAgent(wrong))
    assert summary.accuracy == 0
    for r in summary.records:
        assert r.failure_category in FAILURE_CATEGORIES
        assert r.score_detail


# --------------------------------------------------------------------------------------
# Reliability: crashes and hangs become records, never abort the eval
# --------------------------------------------------------------------------------------
class ScriptAgent:
    """Minimal Agent: behaviour chosen per question."""

    def __init__(self, behaviours):
        self.behaviours = behaviours

    def run(self, question: str) -> AgentResult:
        b = self.behaviours[question]
        if b == "crash":
            raise RuntimeError("boom")
        if b == "hang":
            time.sleep(2)
        return AgentResult([(1,)], "SELECT 1", [], 0, 0, 0, 0.0, 5)


@needs_db
def test_crash_and_hang_are_recorded_and_the_eval_continues(store, monkeypatch):
    monkeypatch.setattr(runner, "HARD_TIMEOUT_GRACE_S", 0.2)
    tasks = golden.load_tasks()[:3]
    agent = ScriptAgent(
        {tasks[0].question: "crash", tasks[1].question: "hang", tasks[2].question: "ok"}
    )
    summary = run(config(timeout_s=0), tasks, store, agent, concurrency=3)
    by_id = {r.task_id: r for r in summary.records}
    assert by_id[tasks[0].id].failure_category == "agent_error"
    assert "boom" in by_id[tasks[0].id].error
    assert by_id[tasks[1].id].failure_category == "timeout"
    assert by_id[tasks[2].id].stop_reason == "answered"  # unaffected by its neighbours
    assert store.count() == 3


@needs_db
def test_agent_deadline_stops_a_slow_model():
    def slow(messages, tools, timeout=None):
        time.sleep(0.3)
        return LLMResponse(None, [ToolCallRequest("c", "list_tables", {}, "{}")])

    result = TextToSQLAgent(slow, timeout_s=0.5).run("q")
    assert result.stop_reason == "timeout"
    assert result.n_llm_calls == 2  # stopped at the deadline, long before max_iters


@needs_db
def test_cached_calls_report_nominal_latency_and_zero_spend():
    def cached(messages, tools, timeout=None):
        if messages[-1]["role"] == "tool":
            return LLMResponse("Done.", cost_usd=0.01, latency_ms=4000, cached=True)
        args = {"query": "SELECT 1"}
        return LLMResponse(
            None,
            [ToolCallRequest("c", "run_sql", args, json.dumps(args))],
            cost_usd=0.02,
            latency_ms=6000,
            cached=True,
        )

    r = TextToSQLAgent(cached).run("q")
    assert (r.n_llm_calls, r.n_cached_calls) == (2, 2)
    assert r.cost_usd == pytest.approx(0.03)  # nominal: what the run cost originally
    assert r.spent_usd == 0.0  # actual: nothing spent this time
    assert r.latency_ms >= 10_000  # nominal latency replayed, not ~0ms


# --------------------------------------------------------------------------------------
# Config + task selection
# --------------------------------------------------------------------------------------
def test_all_shipped_configs_load():
    for path in sorted((db.DATA_DIR.parent / "configs").glob("*.yaml")):
        cfg = load_config(path)
        assert cfg.id == path.stem


def test_load_config_rejects_unknown_keys(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("model: x\nprompt_version: v1\ntemprature: 0\n")
    with pytest.raises(ValueError, match="unknown config keys"):
        load_config(bad)


def test_select_tasks():
    tasks = golden.load_tasks()
    assert [t.id for t in runner.select_tasks(tasks, "m03,e01", None)] == ["m03", "e01"]
    assert len(runner.select_tasks(tasks, None, 5)) == 5
    with pytest.raises(SystemExit):
        runner.select_tasks(tasks, "nope", None)
