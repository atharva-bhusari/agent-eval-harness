"""The CI smoke eval passes, and actually fails when an outcome changes."""

import pytest

from eval_harness import db, smoke
from eval_harness.types import RunRecord

needs_db = pytest.mark.skipif(
    not db.DEFAULT_DB_PATH.exists(), reason="run `make db` to build data/northwind.duckdb"
)


@needs_db
def test_smoke_passes_and_writes_a_report(tmp_path, capsys):
    assert smoke.main(["--out", str(tmp_path)]) == 0
    assert "smoke OK" in capsys.readouterr().out
    md = (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    assert md.startswith("# Smoke eval") and "`schema_hallucination`" in md


def _rec(task_id, success, category, retries=0):
    return RunRecord(
        run_id=task_id,
        task_id=task_id,
        config_id="smoke",
        model="m",
        prompt_version="v1",
        success=success,
        failure_category=category,
        latency_ms=1,
        prompt_tokens=0,
        completion_tokens=0,
        cost_usd=0.0,
        n_tool_calls=1,
        n_retries=retries,
        final_sql=None,
        ts="2026-11-21T00:00:00+00:00",
    )


def test_check_flags_every_kind_of_regression():
    good = [
        _rec("e01", True, None),
        _rec("e05", True, None),
        _rec("m03", True, None, retries=1),
        _rec("m08", False, "wrong_result"),
        _rec("h05", False, "schema_hallucination"),
    ]
    assert smoke.check(good) == []
    broken = [
        _rec("e01", True, None),
        _rec("e05", False, "wrong_result"),  # scorer regression
        _rec("m03", True, None, retries=0),  # retry counting regression
        _rec("m08", False, "wrong_result"),
        _rec("h05", False, "other"),  # classifier regression
    ]
    problems = smoke.check(broken)
    assert len(problems) == 3
    assert any(p.startswith("e05") for p in problems)
    assert any("retries" in p for p in problems)
    assert any(p.startswith("h05") for p in problems)
    assert smoke.check(good[:4]) == ["h05: no run recorded"]
