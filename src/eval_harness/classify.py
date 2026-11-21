"""Failure taxonomy classifier: one category from ``FAILURE_CATEGORIES`` per failed run.

PROVISIONAL (Phase 4): rules use only the run's structured signals (stop reason, tool-call
errors, result shape) so every failed run gets a valid category. Phase 5 refines the
rules (DB parse errors -> sql_syntax_error, unknown tables/columns ->
schema_hallucination) and adds the full test suite.

Rules are checked in priority order: how the run *ended* first, then what it returned.
"""

from __future__ import annotations

from eval_harness.agent.sql_tools import REJECTED_PREFIX
from eval_harness.types import FAILURE_CATEGORIES, AgentResult, Task


def classify(task: Task, result: AgentResult) -> str:
    if result.stop_reason == "timeout":
        category = "timeout"
    elif result.stop_reason == "error":
        category = "agent_error"
    elif any(c.error and c.error.startswith(REJECTED_PREFIX) for c in result.tool_calls):
        category = "wrong_tool"  # tried to write / run a non-SELECT
    elif any(c.error and c.error.startswith("unknown tool") for c in result.tool_calls):
        category = "wrong_tool"
    elif result.stop_reason == "max_iters" and result.answer_rows is None:
        category = "max_iterations"
    elif result.answer_rows is not None and not result.answer_rows and task.expected_result:
        category = "empty_result"
    elif result.answer_rows is not None:
        category = "wrong_result"
    else:
        category = "other"  # e.g. finished without ever running a successful query
    assert category in FAILURE_CATEGORIES
    return category
