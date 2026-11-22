"""Report aggregation + Markdown, and a headless smoke test of the Streamlit dashboard."""

from pathlib import Path

import pytest

from eval_harness import golden, report
from eval_harness.runner import build_record
from eval_harness.store import Store
from eval_harness.types import AgentResult, Config, ToolCall

DASHBOARD = Path(report.__file__).with_name("dashboard.py")
TASKS = {t.id: t for t in golden.load_tasks()}
A = Config(id="cfg_a", model="ollama/x", prompt_version="v1")
B = Config(id="cfg_b", model="openai/y", prompt_version="v2")


def result(rows, calls=(), stop="answered", cost=0.0, latency=1000) -> AgentResult:
    return AgentResult(
        answer_rows=rows,
        final_sql="SELECT 1" if rows is not None else None,
        tool_calls=list(calls),
        n_retries=1,
        prompt_tokens=100,
        completion_tokens=10,
        cost_usd=cost,
        latency_ms=latency,
        stop_reason=stop,
        messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}],
    )


def right(tid, **kw):
    return result(
        list(TASKS[tid].expected_result), [ToolCall("run_sql", {"query": "x"}, True)], **kw
    )


SCHEMA_ERR = 'Binder Error: Referenced column "nme" not found in FROM clause!'


@pytest.fixture
def store_path(tmp_path):
    path = tmp_path / "runs.sqlite"
    runs = [
        (A, "e01", right("e01")),
        (A, "m01", result([("nope",)], [ToolCall("run_sql", {"query": "x"}, True)])),
        (
            A,
            "h01",
            result(
                None, [ToolCall("run_sql", {"query": "x"}, False, SCHEMA_ERR)], stop="max_iters"
            ),
        ),
        (B, "e01", right("e01", cost=0.002, latency=3000)),
        (B, "m01", right("m01", cost=0.004, latency=5000)),
        (B, "h01", result([], [ToolCall("run_sql", {"query": "x"}, True)], cost=0.003)),
    ]
    with Store(path) as s:
        for cfg, tid, r in runs:
            s.upsert(build_record(TASKS[tid], cfg, r), r.tool_calls)
    return path


# ---- aggregation ---------------------------------------------------------------------
def test_wilson_interval():
    lo, hi = report.wilson_interval(40, 50)
    assert (round(lo, 3), round(hi, 3)) == (0.670, 0.888)  # textbook value for 80% of 50
    assert report.wilson_interval(0, 10)[0] == 0.0
    assert report.wilson_interval(10, 10)[1] == pytest.approx(1.0)
    assert report.wilson_interval(0, 0) == (0.0, 0.0)


def test_config_summary(store_path):
    s = report.config_summary(report.load_runs(store_path), n_tasks=50)
    assert list(s["config_id"]) == ["cfg_b", "cfg_a"]  # sorted by accuracy
    b, a = s.iloc[0], s.iloc[1]
    assert (b.successes, b.n, round(b.accuracy, 3)) == (2, 3, 0.667)
    assert round(b.mean_cost_usd, 4) == 0.003 and round(b.mean_latency_s, 2) == 3.0
    assert b.failures == "empty_result 1"
    assert sorted(a.failures.split(", ")) == ["schema_hallucination 1", "wrong_result 1"]
    assert b.n_tasks == 50 and b.ci_low < b.accuracy < b.ci_high


def test_accuracy_by_difficulty_is_in_spec_order(store_path):
    d = report.accuracy_by_difficulty(report.load_runs(store_path))
    assert list(d[d.config_id == "cfg_a"]["difficulty"]) == ["easy", "medium", "hard"]
    assert list(d[d.config_id == "cfg_a"]["accuracy"]) == [1.0, 0.0, 0.0]


def test_failure_counts_follow_taxonomy_order(store_path):
    fc = report.failure_counts(report.load_runs(store_path))
    assert list(fc.index) == ["schema_hallucination", "wrong_result", "empty_result"]
    assert fc.at["empty_result", "cfg_b"] == 1 and fc.at["empty_result", "cfg_a"] == 0


def test_config_filter(store_path):
    assert set(report.load_runs(store_path, ["cfg_a"])["config_id"]) == {"cfg_a"}


# ---- markdown ------------------------------------------------------------------------
def test_markdown_report(store_path, tmp_path):
    out = report.write_report(store_path, tmp_path / "REPORT.md", n_tasks=50)
    md = out.read_text(encoding="utf-8")
    assert "| `cfg_b` | `openai/y` | v2 | 3/50 ⚠️ | **66.7%** (2/3) |" in md
    assert "partial run" in md
    assert "| easy | 100% (1/1) | 100% (1/1) |" in md
    assert "| `schema_hallucination` | 0 | 1 |" in md
    assert "run_sql: Binder Error" in md  # classifier reason shown for no-answer failures
    assert "row count differs: expected" in md  # scorer detail shown for wrong answers
    assert (
        report.write_report(store_path, tmp_path / "again.md", n_tasks=50).read_text(
            encoding="utf-8"
        )
        == md
    )  # deterministic: no timestamps


def test_markdown_escapes_pipes():
    table = report._md_table(["a"], [["x | y"]], "l")
    assert "x \\| y" in table


def test_empty_store(tmp_path):
    md = report.write_report(tmp_path / "empty.sqlite", tmp_path / "R.md", n_tasks=50)
    assert "No runs in the store yet" in md.read_text(encoding="utf-8")


# ---- dashboard -----------------------------------------------------------------------
def _app(monkeypatch, path):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("EVAL_HARNESS_STORE", str(path))
    return AppTest.from_file(str(DASHBOARD), default_timeout=60)


def test_dashboard_renders_every_section(store_path, monkeypatch):
    at = _app(monkeypatch, store_path).run()
    assert not at.exception, at.exception
    headers = [h.value for h in at.subheader]
    assert headers == [
        "Leaderboard",
        "Accuracy by difficulty",
        "Failure taxonomy",
        "Tasks × configs",
        "Drill-down",
    ]
    # drill-down defaults to cfg_a's first failed task (m01, golden order) and shows why
    assert at.selectbox[0].value == "cfg_a" and at.selectbox[1].value == "m01"
    assert "wrong_result" in at.error[0].value


def test_dashboard_drill_down_can_switch_task_and_config(store_path, monkeypatch):
    at = _app(monkeypatch, store_path).run()
    at.selectbox[0].set_value("cfg_b").run()  # config
    at.selectbox[1].set_value("m01").run()  # task
    assert not at.exception, at.exception
    assert at.success and "Passed" in at.success[0].value


def test_dashboard_handles_empty_and_missing_store(tmp_path, monkeypatch):
    with Store(tmp_path / "empty.sqlite"):
        pass
    at = _app(monkeypatch, tmp_path / "empty.sqlite").run()
    assert not at.exception and "no runs yet" in at.info[0].value
    at = _app(monkeypatch, tmp_path / "missing.sqlite").run()
    assert not at.exception and "Store not found" in at.error[0].value
