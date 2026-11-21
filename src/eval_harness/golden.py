"""Golden task set: load, snapshot, and verify ``data/tasks.yaml``.

Workflow for adding tasks:
1. Add an entry with ``question``, ``expected_sql``, ``difficulty``, ``tags`` (and
   ``ordered: true`` only if row order is part of the answer). Leave out ``expected_result``.
2. ``python -m eval_harness.golden snapshot``: runs the SQL and stores its rows.
3. Hand-check the new rows, then ``python -m eval_harness.golden verify`` (``make verify-tasks``).

``verify`` re-executes every ``expected_sql`` and checks it reproduces the stored
``expected_result``, plus lint rules that keep the set well-formed. Scoring only ever uses
``expected_result``; the SQL exists to prove those rows are reproducible.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from eval_harness import db
from eval_harness.types import DIFFICULTIES, Task

TASKS_PATH = db.DATA_DIR / "tasks.yaml"

KNOWN_TAGS = frozenset(
    {
        "aggregation",
        "date",
        "filter",
        "having",
        "join",
        "null_handling",
        "ranking",
        "self_join",
        "subquery",
        "window",
    }
)
MAX_RESULT_ROWS = 25  # keeps answers reviewable and the YAML readable
FIELD_ORDER = ("id", "difficulty", "tags", "ordered", "question", "expected_sql", "expected_result")


# --------------------------------------------------------------------------------------
# Load / save
# --------------------------------------------------------------------------------------
def _read(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data.get("tasks"), list):
        raise ValueError(f"{path}: expected a top-level 'tasks' list")
    return data


def _to_task(raw: dict[str, Any]) -> Task:
    if "expected_result" not in raw:
        raise ValueError(f"task {raw.get('id')!r} has no expected_result; run `snapshot` first")
    return Task(
        id=str(raw["id"]),
        question=raw["question"],
        expected_result=[tuple(r) for r in raw["expected_result"]],
        expected_sql=raw.get("expected_sql"),
        difficulty=raw.get("difficulty", "medium"),
        tags=list(raw.get("tags", [])),
        ordered=bool(raw.get("ordered", False)),
    )


def load_tasks(path: Path | str = TASKS_PATH) -> list[Task]:
    return [_to_task(raw) for raw in _read(Path(path))["tasks"]]


class _FlowList(list):
    """Rendered inline in YAML: ``[a, b]``."""


yaml.SafeDumper.add_representer(
    _FlowList, lambda d, v: d.represent_sequence("tag:yaml.org,2002:seq", v, flow_style=True)
)

_HEADER = """\
# Golden task set for the Northwind text-to-SQL agent.
# expected_result is the SOURCE OF TRUTH for scoring (order-insensitive unless ordered: true).
# expected_sql is reference-only: `make verify-tasks` re-runs it to prove the rows reproduce.
# Add tasks without expected_result, then run `python -m eval_harness.golden snapshot`.
"""


def dump_tasks(raw_tasks: list[dict[str, Any]], path: Path | str = TASKS_PATH) -> None:
    out = []
    for raw in raw_tasks:
        item = {k: raw[k] for k in FIELD_ORDER if k in raw}
        item.update({k: v for k, v in raw.items() if k not in FIELD_ORDER})
        item["tags"] = _FlowList(item.get("tags", []))
        if "expected_result" in item:
            item["expected_result"] = [_FlowList(r) for r in item["expected_result"]]
        out.append(item)
    body = yaml.safe_dump(
        {"version": 1, "tasks": out}, sort_keys=False, allow_unicode=True, width=100
    )
    body = re.sub(r"\n(- id:)", r"\n\n\1", body)  # blank line between tasks
    Path(path).write_text(_HEADER + body, encoding="utf-8")


# --------------------------------------------------------------------------------------
# Snapshot / verify
# --------------------------------------------------------------------------------------
def _plain(v: Any) -> Any:
    """DuckDB values -> YAML-safe scalars (dates/decimals become str/float)."""
    if v is None or isinstance(v, bool | int | float | str):
        return v
    return db.normalize_value(v)


def snapshot(path: Path | str = TASKS_PATH, refresh: bool = False, con=None) -> list[str]:
    """Fill in ``expected_result`` for tasks that lack it (all tasks if ``refresh``).
    Returns the ids that were (re)snapshotted."""
    path = Path(path)
    data = _read(path)
    own = con is None
    con = con or db.connect()
    try:
        changed = []
        for raw in data["tasks"]:
            if refresh or "expected_result" not in raw:
                rows = db.run_query(con, raw["expected_sql"]).rows
                raw["expected_result"] = [[_plain(v) for v in r] for r in rows]
                changed.append(str(raw["id"]))
    finally:
        if own:
            con.close()
    dump_tasks(data["tasks"], path)
    return changed


@dataclass
class Issue:
    task_id: str
    problem: str


def lint(raw_tasks: list[dict[str, Any]]) -> list[Issue]:
    """Structural rules that keep the golden set well-formed (no DB needed)."""
    issues: list[Issue] = []
    counts = Counter(str(r.get("id")) for r in raw_tasks)
    for tid, n in counts.items():
        if n > 1:
            issues.append(Issue(tid, f"duplicate id ({n}x)"))
    for r in raw_tasks:
        tid = str(r.get("id"))
        for key in ("question", "expected_sql", "expected_result"):
            if not r.get(key) and r.get(key) != []:
                issues.append(Issue(tid, f"missing {key}"))
        if r.get("difficulty") not in DIFFICULTIES:
            issues.append(Issue(tid, f"difficulty must be one of {DIFFICULTIES}"))
        unknown = set(r.get("tags", [])) - KNOWN_TAGS
        if unknown:
            issues.append(Issue(tid, f"unknown tags {sorted(unknown)}"))
        if not r.get("tags"):
            issues.append(Issue(tid, "no tags"))
        rows = r.get("expected_result")
        if rows == []:
            issues.append(
                Issue(tid, "empty expected_result (ambiguous: can't tell right from broken)")
            )
        elif rows and len(rows) > MAX_RESULT_ROWS:
            issues.append(Issue(tid, f"{len(rows)} rows > {MAX_RESULT_ROWS}; narrow the question"))
        if r.get("ordered") and "order by" not in str(r.get("expected_sql", "")).lower():
            issues.append(Issue(tid, "ordered: true but expected_sql has no ORDER BY"))
    return issues


def verify(path: Path | str = TASKS_PATH, con=None) -> list[Issue]:
    """Lint the file, then re-run every expected_sql against the stored rows."""
    raw_tasks = _read(Path(path))["tasks"]
    issues = lint(raw_tasks)
    own = con is None
    con = con or db.connect()
    try:
        for raw in raw_tasks:
            if "expected_result" not in raw or not raw.get("expected_sql"):
                continue
            tid = str(raw["id"])
            try:
                actual = db.run_query(con, raw["expected_sql"]).rows
            except Exception as exc:
                issues.append(Issue(tid, f"expected_sql failed: {exc}"))
                continue
            if not db.results_match(
                raw["expected_result"],
                actual,
                ordered=bool(raw.get("ordered")),
                ignore_column_order=False,  # the reference must match exactly, columns too
            ):
                issues.append(Issue(tid, "expected_sql does not reproduce expected_result"))
    finally:
        if own:
            con.close()
    return issues


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m eval_harness.golden")
    p.add_argument("cmd", choices=["verify", "snapshot", "stats"])
    p.add_argument("--path", type=Path, default=TASKS_PATH)
    p.add_argument("--refresh", action="store_true", help="snapshot: re-snapshot every task")
    args = p.parse_args(argv)

    if args.cmd == "snapshot":
        changed = snapshot(args.path, refresh=args.refresh)
        print(f"snapshotted {len(changed)} task(s): {', '.join(changed) or '-'}")
        print("Hand-check the new expected_result rows, then run verify.")
        return 0

    tasks = load_tasks(args.path)
    if args.cmd == "stats":
        print(f"{len(tasks)} tasks")
        print("difficulty:", dict(Counter(t.difficulty for t in tasks)))
        print("tags:", dict(Counter(tag for t in tasks for tag in t.tags).most_common()))
        print("ordered:", sum(t.ordered for t in tasks))
        return 0

    issues = verify(args.path)
    for i in issues:
        print(f"FAIL {i.task_id}: {i.problem}")
    print(f"{len(tasks) - len({i.task_id for i in issues})}/{len(tasks)} tasks verified")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
