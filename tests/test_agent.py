"""Agent tests with a scripted LLM: deterministic, offline, no API key (CI-safe)."""

from __future__ import annotations

import json

import duckdb
import pytest

from eval_harness import db
from eval_harness.agent.prompts import compact_schema, system_prompt
from eval_harness.agent.sql_tools import SqlRejected, SqlTools, validate_select
from eval_harness.agent.text_to_sql import (
    NUDGE_MESSAGE,
    TextToSQLAgent,
    count_retries,
    recover_text_tool_calls,
)
from eval_harness.llm import LLMResponse, ToolCallRequest, litellm_model_id
from eval_harness.types import Agent, ToolCall

# --------------------------------------------------------------------------------------
# Read-only guard (no DB needed)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    ["SELECT 1", "select * from orders", "WITH t AS (SELECT 1 AS x) SELECT x FROM t", "SELECT 1;"],
)
def test_guard_allows_single_select(sql):
    validate_select(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM orders",
        "DROP TABLE orders",
        "INSERT INTO region VALUES (9, 'x')",
        "UPDATE products SET unit_price = 0",
        "CREATE TABLE x (a INT)",
        "SELECT 1; DROP TABLE orders",  # piggy-backed statement
        "ATTACH 'other.db'",
        "COPY orders TO 'out.csv'",
    ],
)
def test_guard_rejects_non_select(sql):
    with pytest.raises(SqlRejected):
        validate_select(sql)


def test_guard_lets_syntax_errors_through_as_parser_errors():
    with pytest.raises(duckdb.ParserException):
        validate_select("SELEC 1")


def test_count_retries_only_counts_corrected_sql_failures():
    calls = [
        ToolCall("list_tables", {}, True),
        ToolCall("run_sql", {}, False),  # failed, then retried -> 1
        ToolCall("get_schema", {}, False),  # not SQL: ignored
        ToolCall("dry_run", {}, False),  # failed, then retried -> 2
        ToolCall("run_sql", {}, True),
        ToolCall("run_sql", {}, False),  # final failure, never retried -> not counted
    ]
    assert count_retries(calls) == 2


def test_ollama_model_ids_route_to_chat_endpoint():
    assert litellm_model_id("ollama/llama3.1:8b") == "ollama_chat/llama3.1:8b"
    assert litellm_model_id("openai/gpt-4o-mini") == "openai/gpt-4o-mini"


# --------------------------------------------------------------------------------------
# Agent loop with a scripted LLM (needs the built DB)
# --------------------------------------------------------------------------------------
needs_db = pytest.mark.skipif(
    not db.DEFAULT_DB_PATH.exists(), reason="run `make db` to build data/northwind.duckdb"
)


def call(name: str, **args) -> ToolCallRequest:
    raw = json.dumps(args)
    return ToolCallRequest(id=f"c_{name}_{len(raw)}", name=name, args=args, raw_args=raw)


class ScriptedLLM:
    """Replays a fixed list of responses and records what the agent sent each turn."""

    def __init__(self, *responses: LLMResponse | Exception):
        self.responses = list(responses)
        self.seen: list[list[dict]] = []

    def __call__(self, messages, tools):
        self.seen.append(list(messages))
        nxt = self.responses.pop(0) if self.responses else LLMResponse(content="done")
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def tools_turn(*calls: ToolCallRequest) -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=list(calls),
        prompt_tokens=100,
        completion_tokens=10,
        cost_usd=0.001,
    )


def final(text: str = "Done.") -> LLMResponse:
    return LLMResponse(content=text, prompt_tokens=100, completion_tokens=5, cost_usd=0.001)


@needs_db
def test_agent_satisfies_protocol():
    assert isinstance(TextToSQLAgent(ScriptedLLM()), Agent)


@needs_db
def test_happy_path_returns_rows_sql_and_usage():
    sql = "SELECT COUNT(*) FROM categories"
    llm = ScriptedLLM(
        tools_turn(call("list_tables")),
        tools_turn(call("dry_run", query=sql)),
        tools_turn(call("run_sql", query=sql)),
        final("There are 8 categories."),
    )
    result = TextToSQLAgent(llm).run("How many categories are there?")

    assert result.stop_reason == "answered"
    assert result.answer_rows == [(8,)]
    assert result.final_sql == sql
    assert result.n_retries == 0
    assert [c.name for c in result.tool_calls] == ["list_tables", "dry_run", "run_sql"]
    assert all(c.ok for c in result.tool_calls)
    assert (result.prompt_tokens, result.completion_tokens) == (400, 35)
    assert result.cost_usd == pytest.approx(0.004)
    assert result.error is None


@needs_db
def test_self_correction_feeds_db_error_back_and_retries():
    bad = "SELECT COUNT(*) FROM invoices"  # hallucinated table
    good = "SELECT COUNT(*) FROM orders"
    llm = ScriptedLLM(
        tools_turn(call("run_sql", query=bad)), tools_turn(call("run_sql", query=good)), final()
    )
    result = TextToSQLAgent(llm).run("How many orders?")

    # The DB error was shown to the model before its second attempt.
    tool_msg = llm.seen[1][-1]
    assert tool_msg["role"] == "tool" and "invoices" in tool_msg["content"]
    assert "ERROR" in tool_msg["content"]

    assert result.n_retries == 1
    assert result.answer_rows == [(830,)]
    assert result.final_sql == good
    assert [c.ok for c in result.tool_calls] == [False, True]


@needs_db
def test_non_select_is_rejected_and_db_unchanged():
    llm = ScriptedLLM(tools_turn(call("run_sql", query="DELETE FROM orders")), final())
    result = TextToSQLAgent(llm).run("Delete all orders")

    (tc,) = result.tool_calls
    assert not tc.ok and tc.error.startswith("rejected:")
    assert result.answer_rows is None
    with db.connect() as con:
        assert db.run_query(con, "SELECT COUNT(*) FROM orders").rows == [(830,)]


@needs_db
def test_last_successful_run_sql_is_the_answer():
    llm = ScriptedLLM(
        tools_turn(call("run_sql", query="SELECT DISTINCT country FROM customers LIMIT 3")),
        tools_turn(call("run_sql", query="SELECT COUNT(*) FROM customers")),
        tools_turn(call("run_sql", query="SELECT nope FROM customers")),  # fails afterwards
        final(),
    )
    result = TextToSQLAgent(llm).run("How many customers?")
    assert result.answer_rows == [(91,)]
    assert result.final_sql == "SELECT COUNT(*) FROM customers"


@needs_db
def test_unknown_tool_and_bad_args_are_recorded_not_raised():
    llm = ScriptedLLM(
        tools_turn(
            call("drop_database"),
            ToolCallRequest(id="x", name="run_sql", args=None, raw_args="{not json"),
            call("get_schema", tbl="orders"),  # wrong argument name
            call("get_schema", table="invoices"),  # unknown table
        ),
        final(),
    )
    result = TextToSQLAgent(llm).run("q")
    assert [c.ok for c in result.tool_calls] == [False, False, False, False]
    assert "unknown tool" in result.tool_calls[0].error
    assert "not valid JSON" in result.tool_calls[1].error
    assert "bad arguments" in result.tool_calls[2].error
    assert "unknown table" in result.tool_calls[3].error
    assert result.stop_reason == "answered"


@needs_db
def test_max_iters_stops_a_looping_model():
    looping = [tools_turn(call("list_tables")) for _ in range(10)]
    result = TextToSQLAgent(ScriptedLLM(*looping), max_iters=3).run("q")
    assert result.stop_reason == "max_iters"
    assert len(result.tool_calls) == 3
    assert result.answer_rows is None


@needs_db
def test_llm_exception_is_captured_with_partial_trace():
    llm = ScriptedLLM(tools_turn(call("list_tables")), TimeoutError("provider timed out"))
    result = TextToSQLAgent(llm).run("q")
    assert result.stop_reason == "error"
    assert "TimeoutError" in result.error
    assert [c.name for c in result.tool_calls] == ["list_tables"]  # trace kept


@needs_db
def test_tools_output_is_truncated_for_the_model_but_not_for_scoring():
    with db.connect() as con:
        out = SqlTools(con.cursor()).run_sql("SELECT order_id FROM orders")
    assert len(out.rows) == 830
    assert "830 row(s) (showing first 20)" in out.text


@needs_db
def test_schema_tool_shows_keys_for_join_paths():
    with db.connect() as con:
        out = SqlTools(con.cursor()).get_schema("orders")
    assert "customer_id VARCHAR  -- references customers.customer_id" in out.text


@needs_db
def test_prompt_versions():
    with db.connect() as con:
        v1, v2 = system_prompt("v1", con), system_prompt("v2", con)
        schema = compact_schema(con)
        with pytest.raises(ValueError):
            system_prompt("v9", con)
    assert "list_tables" in v1 and "orders:" not in v1
    assert schema in v2  # v2 embeds the schema up front
    assert "order_details: order_id pk -> orders.order_id" in schema


# --------------------------------------------------------------------------------------
# Fix 1: tool calls written as JSON text
# --------------------------------------------------------------------------------------


def test_recover_llama_style_text_tool_call():
    content = (
        'Let\'s find them.\n\n{"name": "run_sql", "parameters": {"query": "SELECT * FROM orders"}}'
    )
    (c,) = recover_text_tool_calls(content)
    assert (c.name, c.args) == ("run_sql", {"query": "SELECT * FROM orders"})


def test_recover_accepts_arguments_key_and_string_args_and_fences():
    content = (
        '```json\n{"name": "get_schema", "arguments": "{\\"table\\": \\"orders\\"}"}\n```\n'
        'then {"name": "list_tables", "parameters": {}}'
    )
    calls = recover_text_tool_calls(content)
    assert [(c.name, c.args) for c in calls] == [
        ("get_schema", {"table": "orders"}),
        ("list_tables", {}),
    ]


@pytest.mark.parametrize(
    "content",
    [
        None,
        "There are 8 categories.",
        "SELECT COUNT(*) FROM products",  # SQL as text is handled by the nudge, not here
        '{"name": "drop_database", "parameters": {}}',  # not one of our tools
        '{"result": {"rows": 3}}',  # arbitrary JSON
        '{"name": "run_sql", "parameters": ',  # truncated
    ],
)
def test_recover_ignores_non_tool_text(content):
    assert recover_text_tool_calls(content) == []


@needs_db
def test_text_tool_call_is_executed_and_counted():
    sql = "SELECT COUNT(*) FROM categories"
    text = json.dumps({"name": "run_sql", "parameters": {"query": sql}})
    llm = ScriptedLLM(LLMResponse(content=f"Running it:\n{text}"), final("8."))
    result = TextToSQLAgent(llm).run("How many categories?")
    assert result.answer_rows == [(8,)]
    assert result.n_recovered_calls == 1
    assert result.n_nudges == 0


# --------------------------------------------------------------------------------------
# Fix 2: nudge when the model tries to finish without running a query
# --------------------------------------------------------------------------------------


@needs_db
def test_nudge_when_model_writes_sql_as_text():
    sql = "SELECT COUNT(*) FROM products WHERE discontinued = 1"
    llm = ScriptedLLM(
        LLMResponse(content=sql),  # SQL as plain text, no tool call
        tools_turn(call("run_sql", query=sql)),
        final("Done."),
    )
    result = TextToSQLAgent(llm).run("How many products are discontinued?")

    assert llm.seen[1][-1] == {"role": "user", "content": NUDGE_MESSAGE}
    assert result.n_nudges == 1
    with db.connect() as con:
        assert result.answer_rows == db.run_query(con, sql).rows
    assert result.stop_reason == "answered"


@needs_db
def test_nudge_happens_at_most_once():
    llm = ScriptedLLM(LLMResponse(content="I think 8."), LLMResponse(content="Still 8."))
    result = TextToSQLAgent(llm).run("How many categories?")
    assert result.n_nudges == 1
    assert result.stop_reason == "answered"
    assert result.answer_rows is None


@needs_db
def test_no_nudge_once_a_query_has_succeeded():
    sql = "SELECT COUNT(*) FROM categories"
    llm = ScriptedLLM(tools_turn(call("run_sql", query=sql)), final("8."))
    result = TextToSQLAgent(llm).run("How many categories?")
    assert result.n_nudges == 0
    assert len(llm.seen) == 2
