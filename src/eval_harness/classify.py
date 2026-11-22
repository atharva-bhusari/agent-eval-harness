"""Failure taxonomy classifier: exactly one category from ``FAILURE_CATEGORIES`` per failed run.

Rules use only structured signals recorded during the run (stop reason, the ordered tool
calls with their DB error text, and the shape of the answer), so classification is
deterministic, free, and can be re-run over stored results without calling a model.

Priority (first match wins). The category names the *root cause* of the failure:

1. ``timeout``        the run hit its wall-clock budget (how it ended trumps everything:
                      a slow model and a confused model look alike from outside).
2. ``agent_error``    uncaught exception in the agent or the LLM client.
3. ``wrong_tool``     the agent tried to write / run DDL / stack statements. Safety
                      violations are always surfaced, even if a later answer exists.
4. Answer returned (a successful ``run_sql``) but it is wrong:
   ``empty_result``   0 rows when rows were expected;
   ``wrong_result``   anything else.
5. No answer. Look at the last failed tool call:
   ``schema_hallucination``  it referenced a table/column that does not exist
                              (DuckDB catalog/binder "not found", or get_schema on an
                              unknown table);
   ``sql_syntax_error``      the DB refused the final SQL for any other reason: parse
                              error, unknown function, GROUP BY / type / ambiguity binder
                              errors, runtime errors. The spec calls it "syntax"; in
                              practice it means "invalid SQL that isn't a schema mistake";
   ``wrong_tool``            a non-existent tool, bad arguments, or unparsable arguments.
   Only the *last* SQL attempt counts: errors the agent recovered from are not the cause.
6. ``max_iterations`` the step budget ran out with no DB error to blame (e.g. looping on
                      schema lookups without ever running SQL).
7. ``other``          none of the above, e.g. it replied in prose without running SQL.

A max_iters run whose final SQL failed is labelled by that error (5), not 6: "it hit the
loop limit" is a symptom; "it kept querying a column that doesn't exist" is the cause,
and it points at the fix (here: put the schema in the prompt).

CLI (re-label stored runs after a rule change, no model calls):
    python -m eval_harness.classify reclassify [--store results/runs.sqlite] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from eval_harness.agent.sql_tools import REJECTED_PREFIX
from eval_harness.types import FAILURE_CATEGORIES, AgentResult, Task, ToolCall

SQL_TOOLS = {"dry_run", "run_sql"}

# DuckDB error text that means "this table/column does not exist".
_SCHEMA_PATTERNS = [
    re.compile(r"Catalog Error: Table with name .+ does not exist", re.I),
    re.compile(r"Binder Error: Referenced column .+ not found", re.I),
    re.compile(r"Binder Error: Referenced table .+ not found", re.I),
    re.compile(r"Binder Error: Table .+ does not have a column named", re.I),
    re.compile(r"Binder Error: Values list .+ does not have a column named", re.I),
    re.compile(r"^unknown table ", re.I),  # get_schema('nonexistent')
]
_TOOL_MISUSE_PREFIXES = ("unknown tool", "bad arguments for", "tool arguments were not valid")
_DB_ERROR = re.compile(r"^[A-Z][A-Za-z ]* Error:")  # "Parser Error:", "Binder Error:", ...
MAX_REASON_CHARS = 200


@dataclass(frozen=True)
class Classification:
    category: str
    reason: str  # one line for the report / dashboard drill-down

    def __post_init__(self) -> None:
        if self.category not in FAILURE_CATEGORIES:
            raise ValueError(f"unknown failure category {self.category!r}")


def error_kind(error: str | None) -> str | None:
    """Map one tool error string to a failure category (or None if it isn't an error)."""
    if not error:
        return None
    if error.startswith(REJECTED_PREFIX) or error.startswith(_TOOL_MISUSE_PREFIXES):
        return "wrong_tool"
    if any(p.search(error) for p in _SCHEMA_PATTERNS):
        return "schema_hallucination"
    if _DB_ERROR.match(error):
        return "sql_syntax_error"
    return "other"


def _short(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= MAX_REASON_CHARS else text[: MAX_REASON_CHARS - 3] + "..."


def _last_failed_call(calls: Sequence[ToolCall]) -> ToolCall | None:
    """The error the run ended on. If any SQL was attempted, only the last SQL attempt
    counts (a later successful dry_run means the agent fixed that error)."""
    sql = [c for c in calls if c.name in SQL_TOOLS]
    if sql:
        return None if sql[-1].ok else sql[-1]
    failed = [c for c in calls if not c.ok]
    return failed[-1] if failed else None


def explain(task: Task, result: AgentResult) -> Classification:
    """Classify a failed run and say why. Assumes the run already scored as a failure."""
    calls = result.tool_calls
    if result.stop_reason == "timeout":
        return Classification("timeout", _short(result.error or "exceeded timeout_s"))
    if result.stop_reason == "error":
        return Classification("agent_error", _short(result.error or "agent raised"))

    rejected = next((c for c in calls if c.error and c.error.startswith(REJECTED_PREFIX)), None)
    if rejected is not None:
        query = rejected.args.get("query", "")
        return Classification("wrong_tool", _short(f"attempted a non-SELECT statement: {query}"))

    if result.answer_rows is not None:
        if not result.answer_rows and task.expected_result:
            return Classification("empty_result", f"0 rows, expected {len(task.expected_result)}")
        return Classification("wrong_result", "query ran but the result set differs")

    last = _last_failed_call(calls)
    kind = error_kind(last.error) if last else None
    if kind in ("schema_hallucination", "sql_syntax_error", "wrong_tool"):
        return Classification(kind, _short(f"{last.name}: {last.error}"))
    if result.stop_reason == "max_iters":
        return Classification("max_iterations", f"used all steps ({len(calls)} tool calls)")
    if last is not None:
        return Classification("other", _short(f"{last.name}: {last.error}"))
    if not any(c.name in SQL_TOOLS for c in calls):
        return Classification("other", "finished without running any SQL")
    return Classification("other", "finished without a successful run_sql")


def classify(task: Task, result: AgentResult) -> str:
    return explain(task, result).category


# --------------------------------------------------------------------------------------
# Re-classify stored runs (after a rule change), without calling any model
# --------------------------------------------------------------------------------------
def result_from_store(run: dict, tool_calls: list[dict]) -> AgentResult:
    """Rebuild the parts of an AgentResult the classifier reads from a stored run."""
    answer = (
        None if run["answer_json"] is None else [tuple(r) for r in json.loads(run["answer_json"])]
    )
    calls = [
        ToolCall(
            name=c["name"],
            args=json.loads(c["args_json"]) if c["args_json"] else {},
            ok=bool(c["ok"]),
            error=c["error"],
        )
        for c in tool_calls
    ]
    return AgentResult(
        answer_rows=answer,
        final_sql=run["final_sql"],
        tool_calls=calls,
        n_retries=run["n_retries"],
        prompt_tokens=run["prompt_tokens"],
        completion_tokens=run["completion_tokens"],
        cost_usd=run["cost_usd"],
        latency_ms=run["latency_ms"],
        error=run["error"],
        stop_reason=run["stop_reason"],
    )


def reclassify(store, tasks: Sequence[Task], dry_run: bool = False) -> list[tuple]:
    """Re-label every failed run in ``store``. Returns (config, task, old, new, reason)."""
    by_id = {t.id: t for t in tasks}
    changes = []
    for run in store.runs():
        if run["success"]:
            continue
        task = by_id.get(run["task_id"])
        if task is None:
            continue  # task removed from the golden set since this run
        verdict = explain(task, result_from_store(run, store.tool_calls(run["run_id"])))
        changes.append(
            (
                run["config_id"],
                run["task_id"],
                run["failure_category"],
                verdict.category,
                verdict.reason,
            )
        )
        if not dry_run:
            store.set_failure(run["run_id"], verdict.category, verdict.reason)
    return changes


def main(argv: list[str] | None = None) -> int:
    from eval_harness import golden
    from eval_harness.store import DEFAULT_STORE_PATH, Store

    p = argparse.ArgumentParser(prog="python -m eval_harness.classify")
    sub = p.add_subparsers(dest="cmd", required=True)
    rc = sub.add_parser("reclassify", help="re-label failed runs in the store")
    rc.add_argument("--store", type=Path, default=DEFAULT_STORE_PATH)
    rc.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    with Store(args.store) as store:
        changes = reclassify(store, golden.load_tasks(), dry_run=args.dry_run)
    for cfg, tid, old, new, reason in changes:
        mark = "  " if old == new else "->"
        print(f"{cfg:<18} {tid:<4} {old or '-':<20} {mark} {new:<20} {reason}")
    n_changed = sum(old != new for _, _, old, new, _ in changes)
    totals = Counter(new for *_, new, _ in changes)
    print(
        f"\n{len(changes)} failed runs, {n_changed} relabelled"
        + (" (dry run, nothing written)" if args.dry_run else "")
    )
    print("categories: " + ", ".join(f"{k}={v}" for k, v in totals.most_common()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
