"""Execution-accuracy scorer: what counts as the same answer, and the failure details."""

import datetime as dt
from decimal import Decimal

import pytest

from eval_harness import golden
from eval_harness.scorers import execution_accuracy
from eval_harness.types import AgentResult, Task


def task(expected, ordered=False) -> Task:
    return Task(id="t", question="q", expected_result=expected, ordered=ordered)


def answer(rows, stop="answered") -> AgentResult:
    return AgentResult(
        answer_rows=rows,
        final_sql="SELECT ...",
        tool_calls=[],
        n_retries=0,
        prompt_tokens=0,
        completion_tokens=0,
        cost_usd=0.0,
        latency_ms=1,
        stop_reason=stop,
    )


def score(expected, rows, ordered=False):
    return execution_accuracy(task(expected, ordered), answer(rows))


# ---- equivalent answers pass ---------------------------------------------------------
@pytest.mark.parametrize(
    "expected, actual",
    [
        ([("a", 1), ("b", 2)], [("b", 2), ("a", 1)]),  # row order ignored by default
        ([("a", 1), ("b", 2)], [(1, "a"), (2, "b")]),  # consistent column permutation
        ([(1.23456,)], [(1.23461,)]),  # rounded to 4 dp
        ([(3,)], [(3.0,)]),  # int vs float
        ([(Decimal("10.50"),)], [(10.5,)]),  # decimal vs float
        ([(None,)], [(float("nan"),)]),  # NULL vs NaN
        ([(0.0,)], [(-0.0,)]),
        ([("1996-07-04",)], [(dt.date(1996, 7, 4),)]),  # date vs ISO string
        ([("x",), ("x",)], [("x",), ("x",)]),
    ],
)
def test_equivalent_results_match(expected, actual):
    s = score(expected, actual)
    assert s.success and s.detail == "result set matches"


def test_column_names_and_aliases_are_irrelevant():
    """Only values are compared: the agent may name/alias columns however it likes."""
    assert score([("ALFKI", 6)], [("ALFKI", 6)]).success


# ---- wrong answers fail, with a useful detail ----------------------------------------
def test_no_result_set():
    s = execution_accuracy(task([(1,)]), answer(None, stop="max_iters"))
    assert not s.success and s.detail == "no result set (stop_reason=max_iters)"


def test_row_count_differs():
    s = score([(1,), (2,)], [(1,)])
    assert not s.success and s.detail == "row count differs: expected 2, got 1"
    assert score([(1,)], []).detail == "row count differs: expected 1, got 0"


def test_multiset_not_set():
    """Duplicates matter: [x, x] is not [x, y] even though the sets overlap."""
    s = score([("x",), ("x",)], [("x",), ("y",)])
    assert not s.success and "different values" in s.detail


def test_extra_column_fails():
    s = score([("a",)], [("a", 1)])
    assert not s.success and s.detail == "column count differs: expected 1, got 2"


def test_different_values_names_a_missing_row():
    s = score([("a", 1), ("b", 2)], [("a", 1), ("b", 3)])
    assert not s.success
    assert s.detail == "same shape, different values; e.g. expected row ('b', 2.0) not found"


def test_beyond_rounding_tolerance_fails():
    assert not score([(1.2345,)], [(1.2347,)]).success


def test_ordered_tasks_check_order():
    expected = [("a", 3), ("b", 2), ("c", 1)]
    assert score(expected, list(expected), ordered=True).success
    s = score(expected, list(reversed(expected)), ordered=True)
    assert not s.success and s.detail == "right rows, wrong order"
    assert score(expected, list(reversed(expected)), ordered=False).success


def test_inconsistent_column_swap_is_not_accepted():
    """Columns may be permuted once for the whole result, not independently per row."""
    assert not score([(1, 2), (3, 4)], [(2, 1), (3, 4)]).success


# ---- the golden set scores itself perfectly ------------------------------------------
def test_every_golden_task_accepts_its_own_expected_result():
    for t in golden.load_tasks():
        assert execution_accuracy(t, answer(list(t.expected_result))).success, t.id
