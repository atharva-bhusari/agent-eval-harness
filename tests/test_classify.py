"""Failure taxonomy classifier: error mapping (against real DuckDB errors), rule priority,
every category reachable, and re-classification of stored runs."""

import pytest

from eval_harness import db
from eval_harness.agent.sql_tools import SqlTools
from eval_harness.classify import (
    Classification,
    classify,
    error_kind,
    explain,
    reclassify,
    result_from_store,
)
from eval_harness.runner import build_record
from eval_harness.store import Store
from eval_harness.types import FAILURE_CATEGORIES, AgentResult, Config, Task, ToolCall

TASK = Task(id="t1", question="q", expected_result=[(1,), (2,)])


def result(calls=(), rows=None, stop="answered", error=None) -> AgentResult:
    return AgentResult(
        answer_rows=rows,
        final_sql=None,
        tool_calls=list(calls),
        n_retries=0,
        prompt_tokens=0,
        completion_tokens=0,
        cost_usd=0.0,
        latency_ms=1,
        error=error,
        stop_reason=stop,
    )


def sql(ok=True, error=None, query="SELECT 1", name="run_sql") -> ToolCall:
    return ToolCall(name, {"query": query}, ok, error)


# ---- error text -> kind, using the real DB's messages --------------------------------
@pytest.fixture(scope="module")
def tools():
    con = db.connect()
    yield SqlTools(con)
    con.close()


@pytest.mark.parametrize(
    "query, kind",
    [
        ("SELEC * FROM orders", "sql_syntax_error"),  # parser
        ("SELECT * FROM order", "sql_syntax_error"),  # reserved word: parser, not schema
        ("SELECT * FROM ordrs", "schema_hallucination"),  # unknown table
        ("SELECT nme FROM products", "schema_hallucination"),  # unknown unqualified column
        ("SELECT o.shipper_id FROM orders o", "schema_hallucination"),  # unknown qualified
        ("SELECT product_name, COUNT(*) FROM products", "sql_syntax_error"),  # GROUP BY
        ("SELECT datediff(order_date, shipped_date) FROM orders", "sql_syntax_error"),  # sig
        ("SELECT foo(1)", "sql_syntax_error"),  # unknown function is not a schema mistake
        ("SELECT * FROM orders WHERE order_date > 5", "sql_syntax_error"),  # type
        (
            "SELECT order_id FROM orders o JOIN order_details d ON o.order_id = d.order_id",
            "sql_syntax_error",
        ),  # ambiguous column: exists, just unqualified
        ("DELETE FROM orders", "wrong_tool"),  # write rejected by the guard
        ("SELECT 1; SELECT 2", "wrong_tool"),  # stacked statements
    ],
)
def test_error_kind_on_real_duckdb_errors(tools, query, kind):
    out = tools.run_sql(query)
    assert not out.ok
    assert error_kind(out.error) == kind, out.error


def test_runtime_error_counts_as_invalid_sql(tools):
    out = tools.run_sql("SELECT CAST('abc' AS INTEGER)")  # passes dry_run, fails at run time
    assert error_kind(out.error) == "sql_syntax_error"


def test_tool_misuse_errors(tools):
    assert error_kind(tools.call("drop_table", {}).error) == "wrong_tool"
    assert error_kind(tools.call("get_schema", {"query": "x"}).error) == "wrong_tool"
    assert error_kind(tools.call("run_sql", None).error) == "wrong_tool"
    assert error_kind(tools.call("get_schema", {"table": "order_lines"}).error) == (
        "schema_hallucination"
    )


def test_error_kind_of_nothing_and_unknown_text():
    assert error_kind(None) is None and error_kind("") is None
    assert error_kind("something odd happened") == "other"


# ---- rule priority -------------------------------------------------------------------
SCHEMA_ERR = 'Binder Error: Referenced column "nme" not found in FROM clause!'
SYNTAX_ERR = 'Parser Error: syntax error at or near "SELEC"'
REJECTED = "rejected: only a single read-only SELECT ..."


def test_timeout_beats_everything():
    r = result([sql(False, REJECTED, "DELETE FROM x")], stop="timeout", error="deadline")
    assert classify(TASK, r) == "timeout"


def test_agent_error():
    assert classify(TASK, result(stop="error", error="RateLimitError")) == "agent_error"


def test_attempted_write_is_flagged_even_with_a_later_answer():
    r = result([sql(False, REJECTED, "DELETE FROM orders"), sql()], rows=[(9,)])
    c = explain(TASK, r)
    assert c.category == "wrong_tool" and "DELETE FROM orders" in c.reason


def test_empty_vs_wrong_result():
    assert classify(TASK, result([sql()], rows=[])) == "empty_result"
    assert classify(TASK, result([sql()], rows=[(9,)])) == "wrong_result"


def test_a_returned_answer_outranks_earlier_sql_errors():
    """The agent recovered from the error and answered: the answer is what was wrong."""
    r = result([sql(False, SCHEMA_ERR), sql()], rows=[(9,)])
    assert classify(TASK, r) == "wrong_result"


@pytest.mark.parametrize(
    "err, expected", [(SCHEMA_ERR, "schema_hallucination"), (SYNTAX_ERR, "sql_syntax_error")]
)
def test_gave_up_after_a_db_error(err, expected):
    c = explain(TASK, result([sql(False, err)]))
    assert c.category == expected and c.reason.startswith("run_sql: ")


def test_only_the_last_sql_attempt_counts():
    r = result([sql(False, SCHEMA_ERR), sql(False, SYNTAX_ERR, name="dry_run")])
    assert classify(TASK, r) == "sql_syntax_error"
    # fixed the error (dry_run passed) but never ran it: no DB error to blame
    r = result([sql(False, SCHEMA_ERR), sql(True, name="dry_run")])
    assert explain(TASK, r) == Classification("other", "finished without a successful run_sql")


def test_max_iters_names_the_root_cause_when_the_db_refused_the_sql():
    """Real case (llama3.1 v1, m03): 11 identical hallucinated-column queries."""
    calls = [sql(False, 'Binder Error: Table "o" does not have a column named "shipper_id"')] * 11
    assert classify(TASK, result(calls, stop="max_iters")) == "schema_hallucination"


def test_max_iters_without_a_db_error():
    calls = [ToolCall("list_tables", {}, True)] * 12
    assert classify(TASK, result(calls, stop="max_iters")) == "max_iterations"


def test_tool_misuse_without_an_answer():
    calls = [ToolCall("list_tables", {"table": "x"}, False, "bad arguments for list_tables: x")]
    assert classify(TASK, result(calls)) == "wrong_tool"


def test_schema_lookup_hallucination_without_sql():
    calls = [
        ToolCall("get_schema", {"table": "order_lines"}, False, "unknown table 'order_lines'.")
    ]
    assert classify(TASK, result(calls)) == "schema_hallucination"


def test_other():
    assert explain(TASK, result()).reason == "finished without running any SQL"
    assert classify(TASK, result([sql(False, "weird failure")])) == "other"


def test_every_category_is_reachable():
    cases = {
        "timeout": result(stop="timeout"),
        "agent_error": result(stop="error"),
        "wrong_tool": result([sql(False, REJECTED, "DROP TABLE x")]),
        "empty_result": result([sql()], rows=[]),
        "wrong_result": result([sql()], rows=[(3,)]),
        "schema_hallucination": result([sql(False, SCHEMA_ERR)]),
        "sql_syntax_error": result([sql(False, SYNTAX_ERR)]),
        "max_iterations": result([ToolCall("list_tables", {}, True)], stop="max_iters"),
        "other": result(),
    }
    assert set(cases) == set(FAILURE_CATEGORIES)
    for expected, r in cases.items():
        assert classify(TASK, r) == expected


def test_reason_is_one_bounded_line():
    long_err = "Parser Error: " + "x " * 500 + "\n second line"
    c = explain(TASK, result([sql(False, long_err)]))
    assert "\n" not in c.reason and len(c.reason) <= 200


def test_classification_rejects_unknown_category():
    with pytest.raises(ValueError):
        Classification("made_up", "")


# ---- stored runs ---------------------------------------------------------------------
CONFIG = Config(id="cfg", model="ollama/x", prompt_version="v1")


def test_build_record_stores_category_and_reason():
    rec = build_record(TASK, CONFIG, result([sql(False, SCHEMA_ERR)]))
    assert rec.failure_category == "schema_hallucination"
    assert rec.failure_detail.startswith("run_sql: Binder Error")
    ok = build_record(TASK, CONFIG, result([sql()], rows=[(1,), (2,)]))
    assert ok.success and ok.failure_category is None and ok.failure_detail is None


def test_stored_run_round_trips_to_the_same_classification(tmp_path):
    r = result([sql(False, SCHEMA_ERR), sql(False, SYNTAX_ERR)], stop="max_iters")
    rec = build_record(TASK, CONFIG, r)
    with Store(tmp_path / "runs.sqlite") as store:
        store.upsert(rec, r.tool_calls)
        run = store.runs()[0]
        rebuilt = result_from_store(run, store.tool_calls(run["run_id"]))
    assert explain(TASK, rebuilt) == explain(TASK, r)


def test_reclassify_updates_old_labels_and_leaves_successes(tmp_path):
    bad = result([sql(False, SCHEMA_ERR)], stop="max_iters")
    good = result([sql()], rows=[(1,), (2,)])
    other_task = Task(id="t2", question="q", expected_result=[(1,), (2,)])
    with Store(tmp_path / "runs.sqlite") as store:
        stale = build_record(TASK, CONFIG, bad)
        stale.failure_category, stale.failure_detail = "max_iterations", None  # old rules
        store.upsert(stale, bad.tool_calls)
        store.upsert(build_record(other_task, CONFIG, good), good.tool_calls)

        preview = reclassify(store, [TASK, other_task], dry_run=True)
        assert [(c[2], c[3]) for c in preview] == [("max_iterations", "schema_hallucination")]
        assert store.runs()[0]["failure_category"] == "max_iterations"  # dry run wrote nothing

        reclassify(store, [TASK, other_task])
        rows = {r["task_id"]: r for r in store.runs()}
    assert rows["t1"]["failure_category"] == "schema_hallucination"
    assert rows["t1"]["failure_detail"].startswith("run_sql:")
    assert rows["t2"]["success"] == 1 and rows["t2"]["failure_category"] is None


def test_set_failure_validates(tmp_path):
    with Store(tmp_path / "runs.sqlite") as store:
        store.upsert(build_record(TASK, CONFIG, result([sql()], rows=[(1,), (2,)])))
        with pytest.raises(ValueError):
            store.set_failure("x", "made_up", None)
        with pytest.raises(KeyError):  # successful runs can't be given a failure label
            store.set_failure(store.runs()[0]["run_id"], "other", None)
