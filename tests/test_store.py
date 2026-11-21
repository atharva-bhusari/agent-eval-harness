"""SQLite store: idempotent upserts, tool-call replacement, additive migration."""

import sqlite3

from eval_harness.store import Store
from eval_harness.types import RunRecord, ToolCall


def record(**overrides) -> RunRecord:
    base = dict(
        run_id="r1",
        task_id="e01",
        config_id="cfg",
        model="m",
        prompt_version="v1",
        success=False,
        failure_category="wrong_result",
        latency_ms=100,
        prompt_tokens=10,
        completion_tokens=2,
        cost_usd=0.01,
        n_tool_calls=2,
        n_retries=0,
        final_sql="SELECT 1",
        ts="2026-11-21T00:00:00+00:00",
    )
    base.update(overrides)
    return RunRecord(**base)


def test_upsert_is_idempotent_and_overwrites(tmp_path):
    with Store(tmp_path / "runs.sqlite") as s:
        s.upsert(
            record(),
            [ToolCall("list_tables", {}, True), ToolCall("run_sql", {"query": "x"}, False, "err")],
        )
        s.upsert(
            record(success=True, failure_category=None, latency_ms=50),
            [ToolCall("run_sql", {}, True)],
        )
        rows = s.runs()
        assert len(rows) == 1
        assert rows[0]["success"] == 1 and rows[0]["failure_category"] is None
        assert rows[0]["latency_ms"] == 50
        calls = s.tool_calls("r1")
        assert [(c["name"], c["ok"]) for c in calls] == [("run_sql", 1)]  # replaced, not appended


def test_distinct_run_ids_coexist_and_filter_by_config(tmp_path):
    with Store(tmp_path / "runs.sqlite") as s:
        s.upsert(record(run_id="a", config_id="x"))
        s.upsert(record(run_id="b", config_id="y"))
        assert s.count() == 2
        assert s.run_ids("x") == {"a"}


def test_data_persists_across_connections(tmp_path):
    path = tmp_path / "runs.sqlite"
    with Store(path) as s:
        s.upsert(record())
    with Store(path) as s:
        assert s.count() == 1


def test_old_table_gets_new_columns(tmp_path):
    path = tmp_path / "runs.sqlite"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE runs (run_id TEXT PRIMARY KEY, task_id TEXT)")
    con.commit()
    con.close()
    with Store(path) as s:
        s.upsert(record())
        assert s.runs()[0]["spent_usd"] == 0.0
