"""Tests for the golden task set and its tooling."""

from collections import Counter

import pytest

from eval_harness import db, golden
from eval_harness.types import Task

needs_db = pytest.mark.skipif(
    not db.DEFAULT_DB_PATH.exists(), reason="run `make db` to build data/northwind.duckdb"
)

# --------------------------------------------------------------------------------------
# The real data/tasks.yaml
# --------------------------------------------------------------------------------------


def test_golden_set_shape():
    tasks = golden.load_tasks()
    assert len(tasks) == 50
    assert all(isinstance(t, Task) for t in tasks)
    assert len({t.id for t in tasks}) == 50
    assert Counter(t.difficulty for t in tasks) == {"easy": 15, "medium": 20, "hard": 15}
    tags = {tag for t in tasks for tag in t.tags}
    assert {"aggregation", "join", "filter", "ranking", "subquery"} <= tags  # spec coverage
    assert tags <= golden.KNOWN_TAGS


def test_golden_set_passes_lint():
    raw = golden._read(golden.TASKS_PATH)["tasks"]
    assert golden.lint(raw) == []


def test_rows_load_as_tuples():
    (e01,) = [t for t in golden.load_tasks() if t.id == "e01"]
    assert e01.expected_result == [(91,)]


@needs_db
def test_golden_set_verifies_against_db():
    assert golden.verify() == []


# --------------------------------------------------------------------------------------
# Tooling on temporary files
# --------------------------------------------------------------------------------------
def _task(**overrides):
    base = {
        "id": "t1",
        "difficulty": "easy",
        "tags": ["aggregation"],
        "ordered": False,
        "question": "How many categories?",
        "expected_sql": "SELECT COUNT(*) FROM categories",
    }
    base.update(overrides)
    return base


@needs_db
def test_snapshot_fills_missing_results_and_round_trips(tmp_path):
    path = tmp_path / "tasks.yaml"
    golden.dump_tasks(
        [
            _task(),
            _task(id="t2", expected_sql="SELECT 'Côte' AS s, 1.5 AS x", expected_result=[["x", 0]]),
        ],
        path,
    )
    changed = golden.snapshot(path)
    assert changed == ["t1"]  # t2 already had a result: untouched without --refresh

    tasks = {t.id: t for t in golden.load_tasks(path)}
    assert tasks["t1"].expected_result == [(8,)]
    assert tasks["t2"].expected_result == [("x", 0)]
    assert golden.snapshot(path, refresh=True) == ["t1", "t2"]
    assert golden.load_tasks(path)[1].expected_result == [("Côte", 1.5)]  # unicode survives


@needs_db
def test_verify_catches_a_wrong_stored_result(tmp_path):
    path = tmp_path / "tasks.yaml"
    golden.dump_tasks([_task(expected_result=[[9]])], path)
    (issue,) = golden.verify(path)
    assert issue.task_id == "t1" and "does not reproduce" in issue.problem


@needs_db
def test_verify_catches_broken_sql(tmp_path):
    path = tmp_path / "tasks.yaml"
    golden.dump_tasks(
        [_task(expected_sql="SELECT nope FROM categories", expected_result=[[8]])], path
    )
    (issue,) = golden.verify(path)
    assert "expected_sql failed" in issue.problem


@pytest.mark.parametrize(
    ("overrides", "problem"),
    [
        ({"difficulty": "extreme"}, "difficulty"),
        ({"tags": ["vibes"]}, "unknown tags"),
        ({"tags": []}, "no tags"),
        ({"expected_result": []}, "empty expected_result"),
        ({"expected_result": [[i] for i in range(30)]}, "narrow the question"),
        ({"ordered": True}, "no ORDER BY"),
        ({"question": ""}, "missing question"),
    ],
)
def test_lint_rules(overrides, problem):
    task = _task(expected_result=[[8]])
    task.update(overrides)
    issues = golden.lint([task])
    assert any(problem in i.problem for i in issues), issues


def test_lint_flags_duplicate_ids():
    issues = golden.lint([_task(expected_result=[[8]]), _task(expected_result=[[8]])])
    assert any("duplicate id" in i.problem for i in issues)


def test_load_refuses_unsnapshotted_tasks(tmp_path):
    path = tmp_path / "tasks.yaml"
    golden.dump_tasks([_task()], path)
    with pytest.raises(ValueError, match="snapshot"):
        golden.load_tasks(path)
