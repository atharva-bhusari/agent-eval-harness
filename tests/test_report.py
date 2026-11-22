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
        "Accuracy vs cost / latency",
        "Summary",
        "Head-to-head (paired)",
        "Tasks × configs",
        "Drill-down",
    ]
    # drill-down defaults to cfg_a's first failed task (m01, golden order) and shows why
    assert at.selectbox[0].value == "cfg_a" and at.selectbox[1].value == "m01"
    assert "wrong_result" in at.error[0].value


def test_dashboard_drill_down_can_switch_task_and_config(store_path, monkeypatch):
    at = _app(monkeypatch, store_path).run()
    at.selectbox[0].set_value("cfg_b").run()  # config
    assert at.selectbox[1].options == ["h01"]  # "failed only" is on by default
    at.toggle(key="failed_only").set_value(False).run()
    at.selectbox[1].set_value("m01").run()  # a passing task
    assert not at.exception, at.exception
    assert at.success and "Passed" in at.success[0].value


def test_dashboard_view_is_restored_from_the_url(store_path, monkeypatch):
    """Shareable links: ?configs=..&run_config=..&task=.. reopens the same view. Task
    options are plain ids, because a URL-bound selectbox stores the displayed label."""
    at = _app(monkeypatch, store_path)
    at.query_params["configs"] = ["cfg_a"]
    at.query_params["run_config"] = "cfg_a"
    at.query_params["task"] = "h01"
    at.run()
    assert not at.exception, at.exception
    assert at.multiselect[0].value == ["cfg_a"]
    assert (at.selectbox[0].value, at.selectbox[1].value) == ("cfg_a", "h01")
    assert "schema_hallucination" in at.error[0].value


def test_dashboard_handles_empty_and_missing_store(tmp_path, monkeypatch):
    with Store(tmp_path / "empty.sqlite"):
        pass
    at = _app(monkeypatch, tmp_path / "empty.sqlite").run()
    assert not at.exception and "no runs yet" in at.info[0].value
    at = _app(monkeypatch, tmp_path / "missing.sqlite").run()
    assert not at.exception and "Store not found" in at.error[0].value


# ---- paired comparison + summary prose (Phase 7) ---------------------------------------
def test_mcnemar_exact():
    assert report.mcnemar_exact(0, 0) == 1.0
    assert report.mcnemar_exact(5, 5) == 1.0
    assert report.mcnemar_exact(0, 6) == pytest.approx(2 / 64)  # 0.03125: significant
    assert report.mcnemar_exact(0, 5) == pytest.approx(2 / 32)  # 0.0625: not quite
    assert report.mcnemar_exact(2, 9) == report.mcnemar_exact(9, 2)  # symmetric


def _df(rows):
    """rows: (config, model, prompt, {task: success})"""
    import pandas as pd

    recs = []
    for cid, model, prompt, results in rows:
        for tid, ok in results.items():
            recs.append(
                dict(
                    config_id=cid,
                    model=model,
                    prompt_version=prompt,
                    task_id=tid,
                    success=ok,
                    difficulty="easy",
                    latency_ms=1000,
                    cost_usd=0.001,
                    n_retries=0,
                    n_tool_calls=1,
                    failure_category=None if ok else "wrong_result",
                    failure_detail=None if ok else "query ran but the result set differs",
                    score_detail="result set matches" if ok else "row count differs",
                )
            )
    return pd.DataFrame(recs)


def test_paired_comparisons_orient_prompt_pairs_and_use_shared_tasks():
    ts = [f"t{i}" for i in range(10)]
    df = _df(
        [
            ("m_v2", "m", "v2", {t: True for t in ts}),
            ("m_v1", "m", "v1", {t: i < 3 for i, t in enumerate(ts)}),
            ("n_v1", "n", "v1", {t: i % 2 == 0 for i, t in enumerate(ts[:4])}),  # partial
        ]
    )
    pairs = report.paired_comparisons(df)
    assert list(pairs["kind"]) == ["prompt", "model", "model + prompt"]
    p = pairs.iloc[0]
    assert (p.a, p.b, p.n, p.a_only, p.b_only) == ("m_v1", "m_v2", 10, 0, 7)
    assert p.delta == pytest.approx(0.7) and p.p_value == pytest.approx(2 / 128)
    assert p.b_only_tasks == "t3, t4, t5, t6, t7, t8, t9"
    model_pair = pairs.iloc[1]
    assert {model_pair.a, model_pair.b} == {"m_v1", "n_v1"} and model_pair.n == 4


def test_summary_prose_calls_a_non_significant_lead_a_tie():
    ts = [f"t{i}" for i in range(50)]
    df = _df(
        [
            ("a", "m", "v1", {t: i < 42 for i, t in enumerate(ts)}),
            ("b", "m", "v2", {t: i < 40 for i, t in enumerate(ts)}),
        ]
    )
    df.loc[df.config_id == "b", "cost_usd"] = 0.0002
    df.loc[df.config_id == "b", "latency_ms"] = 500
    s = report.config_summary(df, 50)
    lines = report.summary_prose(s, report.paired_comparisons(df))
    text = "\n".join(lines)
    assert "**Winner on accuracy: `a`**" in text
    assert "not statistically significant" in text
    assert "`b` is 5.0× cheaper per run" in text and "2.0× faster" in text
    assert "Prompt effect on `m`: `a` → `b` is -4 points" in text


def test_summary_prose_significant_lead_and_no_tradeoff():
    ts = [f"t{i}" for i in range(50)]
    df = _df(
        [
            ("a", "m", "v1", {t: True for t in ts}),
            ("b", "n", "v1", {t: i < 30 for i, t in enumerate(ts)}),
        ]
    )
    df.loc[df.config_id == "b", "cost_usd"] = 0.01
    lines = report.summary_prose(report.config_summary(df, 50), report.paired_comparisons(df))
    assert any("significantly ahead" in x for x in lines)
    assert any("no cost/latency trade-off" in x for x in lines)


def test_partial_configs_cannot_win():
    df = _df(
        [
            ("full", "m", "v1", {f"t{i}": i < 30 for i in range(50)}),
            ("partial", "n", "v1", {"t0": True, "t1": True}),
        ]
    )
    lines = report.summary_prose(report.config_summary(df, 50), report.paired_comparisons(df))
    assert lines[0].startswith("**Winner on accuracy: `full`**")


def test_single_config_report_renders(tmp_path):
    """Regression (found by the smoke eval): one config means no pairs; the summary must
    not crash filtering an empty comparison table."""
    df = _df([("only", "m", "v1", {"t0": True, "t1": False})])
    pairs = report.paired_comparisons(df)
    assert pairs.empty and "kind" in pairs.columns
    md = report.render_markdown(df, n_tasks=2)
    assert "**Winner on accuracy: `only`**" in md and "## Head-to-head" not in md
