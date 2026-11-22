"""SQLite metrics store.

Tables:
- ``runs``: one row per (task, config, prompt_version), PRIMARY KEY ``run_id``. Columns
  follow the spec plus diagnostic extensions (stop reason, nudges, cache stats, answer and
  transcript JSON for the dashboard drill-down).
- ``tool_calls``: one row per tool call, for drill-down.

All writes are upserts keyed by ``run_id``: re-running a task/config overwrites its row
(and replaces its tool calls) in one transaction, so re-runs are idempotent.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, fields
from pathlib import Path

from eval_harness.settings import env, repo_root
from eval_harness.types import FAILURE_CATEGORIES, RunRecord, ToolCall

DEFAULT_STORE_PATH = Path(env("EVAL_HARNESS_STORE", str(repo_root() / "results" / "runs.sqlite")))

_SQL_TYPES = {"str": "TEXT", "int": "INTEGER", "float": "REAL", "bool": "INTEGER"}


def _column_type(annotation: str) -> str:
    base = annotation.replace(" | None", "").strip()
    return _SQL_TYPES.get(base, "TEXT")


RUN_COLUMNS = [f.name for f in fields(RunRecord)]
_RUNS_DDL = (
    "CREATE TABLE IF NOT EXISTS runs (\n  "
    + ",\n  ".join(
        f"{f.name} {_column_type(str(f.type))}" + (" PRIMARY KEY" if f.name == "run_id" else "")
        for f in fields(RunRecord)
    )
    + "\n)"
)
_TOOL_CALLS_DDL = """
CREATE TABLE IF NOT EXISTS tool_calls (
  run_id TEXT NOT NULL,
  seq INTEGER NOT NULL,
  name TEXT,
  ok INTEGER,
  error TEXT,
  args_json TEXT,
  PRIMARY KEY (run_id, seq)
)"""


class Store:
    def __init__(self, path: Path | str = DEFAULT_STORE_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(self.path))
        self.con.row_factory = sqlite3.Row
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute(_RUNS_DDL)
        self.con.execute(_TOOL_CALLS_DDL)
        self._migrate()
        self.con.commit()

    def _migrate(self) -> None:
        """Add columns introduced after the table was created (additive only)."""
        have = {r["name"] for r in self.con.execute("PRAGMA table_info(runs)")}
        for f in fields(RunRecord):
            if f.name not in have:
                self.con.execute(
                    f"ALTER TABLE runs ADD COLUMN {f.name} {_column_type(str(f.type))}"
                )

    def upsert(self, record: RunRecord, tool_calls: list[ToolCall] = ()) -> None:
        row = asdict(record)
        cols = ", ".join(RUN_COLUMNS)
        marks = ", ".join(f":{c}" for c in RUN_COLUMNS)
        updates = ", ".join(f"{c} = excluded.{c}" for c in RUN_COLUMNS if c != "run_id")
        with self.con:  # one transaction: run row + its tool calls
            self.con.execute(
                f"INSERT INTO runs ({cols}) VALUES ({marks}) "
                f"ON CONFLICT(run_id) DO UPDATE SET {updates}",
                row,
            )
            self.con.execute("DELETE FROM tool_calls WHERE run_id = ?", (record.run_id,))
            self.con.executemany(
                "INSERT INTO tool_calls (run_id, seq, name, ok, error, args_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (record.run_id, i, c.name, int(c.ok), c.error, json.dumps(c.args, default=str))
                    for i, c in enumerate(tool_calls)
                ],
            )

    def set_failure(self, run_id: str, category: str, detail: str | None) -> None:
        """Re-label one failed run (used by ``classify reclassify``)."""
        if category not in FAILURE_CATEGORIES:
            raise ValueError(f"unknown failure category {category!r}")
        with self.con:
            cur = self.con.execute(
                "UPDATE runs SET failure_category = ?, failure_detail = ? "
                "WHERE run_id = ? AND success = 0",
                (category, detail, run_id),
            )
        if cur.rowcount != 1:
            raise KeyError(f"no failed run with run_id {run_id!r}")

    def run_ids(self, config_id: str | None = None) -> set[str]:
        sql, args = "SELECT run_id FROM runs", ()
        if config_id:
            sql, args = sql + " WHERE config_id = ?", (config_id,)
        return {r["run_id"] for r in self.con.execute(sql, args)}

    def runs(self, config_id: str | None = None) -> list[dict]:
        sql, args = "SELECT * FROM runs", ()
        if config_id:
            sql, args = sql + " WHERE config_id = ?", (config_id,)
        return [dict(r) for r in self.con.execute(sql + " ORDER BY config_id, task_id", args)]

    def tool_calls(self, run_id: str) -> list[dict]:
        rows = self.con.execute("SELECT * FROM tool_calls WHERE run_id = ? ORDER BY seq", (run_id,))
        return [dict(r) for r in rows]

    def count(self, config_id: str | None = None) -> int:
        return len(self.run_ids(config_id))

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
