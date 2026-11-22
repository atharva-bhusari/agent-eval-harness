"""Scorers.

``execution_accuracy`` is the primary metric: does the agent's result set equal the
golden ``expected_result``? Comparison is by value, via ``db.results_match``: multiset of
rows (ordered for ``task.ordered``), numeric/NULL/date normalization, and column order
ignored if one consistent permutation fits.

An optional subjective ``llm_judge`` for answer-explanation quality is planned (off by
default, cost-bearing) and deliberately kept out of the headline accuracy number.
"""

from __future__ import annotations

from eval_harness.db import normalize_rows, results_match
from eval_harness.types import AgentResult, Score, Task


def execution_accuracy(task: Task, result: AgentResult) -> Score:
    if result.answer_rows is None:
        return Score(False, f"no result set (stop_reason={result.stop_reason})")
    if results_match(task.expected_result, result.answer_rows, ordered=task.ordered):
        return Score(True, "result set matches")

    exp, act = task.expected_result, result.answer_rows
    if len(act) != len(exp):
        return Score(False, f"row count differs: expected {len(exp)}, got {len(act)}")
    exp_w = len(exp[0]) if exp else 0
    act_w = len(act[0]) if act else 0
    if act_w != exp_w:
        return Score(False, f"column count differs: expected {exp_w}, got {act_w}")
    if task.ordered and results_match(exp, act, ordered=False):
        return Score(False, "right rows, wrong order")
    # Same shape, different values: show the first mismatching row to aid debugging.
    got = set(normalize_rows(act))
    missing = [r for r in normalize_rows(exp) if r not in got]
    hint = f"; e.g. expected row {missing[0]!r} not found" if missing else ""
    return Score(False, f"same shape, different values{hint}")
