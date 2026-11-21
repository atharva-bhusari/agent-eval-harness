"""The four tools the text-to-SQL agent may call, plus the read-only guard.

Every tool returns a ``ToolOutcome``: ``text`` is what the model sees; the structured
fields (rows, columns, error) are what the harness records. Tool errors are returned,
never raised, so the agent can read the DB error and self-correct.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import duckdb

from eval_harness import db

MAX_ROWS_SHOWN = 20  # rows echoed back to the model; the full result is kept for scoring
MAX_ERROR_CHARS = 600

REJECTED_PREFIX = "rejected:"
READ_ONLY_MESSAGE = (
    f"{REJECTED_PREFIX} only a single read-only SELECT (or WITH ... SELECT) statement is "
    "allowed. Rewrite the query as one SELECT."
)


class SqlRejected(Exception):
    """The statement parsed fine but is not allowed (write, DDL, multiple statements)."""


def validate_select(sql: str) -> None:
    """Allow exactly one SELECT statement, judged by DuckDB's own parser (not regex).

    Raises ``SqlRejected`` for writes/DDL/multi-statement input and lets
    ``duckdb.ParserException`` through for syntax errors, so the classifier can tell
    "wrote to the DB" (wrong_tool) apart from "invalid SQL" (sql_syntax_error).
    The connection is also read-only at the engine level; this guard exists for clear
    error messages and failure classification, not as the only line of defense.
    """
    statements = duckdb.extract_statements(sql)
    if len(statements) != 1 or statements[0].type != duckdb.StatementType.SELECT:
        raise SqlRejected(READ_ONLY_MESSAGE)


@dataclass
class ToolOutcome:
    ok: bool
    text: str  # shown to the model
    error: str | None = None
    rows: list[tuple] | None = None  # set by a successful run_sql
    columns: list[str] = field(default_factory=list)
    is_sql: bool = False  # dry_run / run_sql: counts toward SQL attempts and retries


def _err(exc: Exception) -> str:
    return " ".join(str(exc).split())[:MAX_ERROR_CHARS]


def _format_rows(columns: list[str], rows: list[tuple]) -> str:
    shown = rows[:MAX_ROWS_SHOWN]
    lines = [f"columns: {columns}", *(repr(tuple(r)) for r in shown)]
    more = f" (showing first {len(shown)})" if len(rows) > len(shown) else ""
    lines.append(f"-- {len(rows)} row(s){more}")
    return "\n".join(lines)


class SqlTools:
    """Tool implementations bound to one (read-only) DuckDB connection/cursor."""

    def __init__(self, con: Any):
        self.con = con
        self._dispatch: dict[str, Callable[..., ToolOutcome]] = {
            "list_tables": self.list_tables,
            "get_schema": self.get_schema,
            "dry_run": self.dry_run,
            "run_sql": self.run_sql,
        }

    @property
    def names(self) -> set[str]:
        return set(self._dispatch)

    def call(self, name: str, args: dict[str, Any] | None) -> ToolOutcome:
        if name not in self._dispatch:
            msg = f"unknown tool {name!r}; available tools: {sorted(self._dispatch)}"
            return ToolOutcome(ok=False, text=f"ERROR: {msg}", error=msg)
        if args is None:
            msg = "tool arguments were not valid JSON"
            return ToolOutcome(ok=False, text=f"ERROR: {msg}", error=msg)
        try:
            return self._dispatch[name](**args)
        except TypeError as exc:  # wrong/missing argument names
            msg = f"bad arguments for {name}: {_err(exc)}"
            return ToolOutcome(ok=False, text=f"ERROR: {msg}", error=msg)

    # ---- tools --------------------------------------------------------------------
    def list_tables(self) -> ToolOutcome:
        return ToolOutcome(ok=True, text=", ".join(db.list_tables(self.con)))

    def get_schema(self, table: str) -> ToolOutcome:
        try:
            cols = db.get_schema(self.con, table)
        except ValueError as exc:
            msg = f"{exc}. Call list_tables to see valid table names."
            return ToolOutcome(ok=False, text=f"ERROR: {msg}", error=msg)
        lines = [f"table {table}:"]
        for c in cols:
            null = "" if c.nullable else " NOT NULL"
            note = f"  -- {c.comment}" if c.comment else ""
            lines.append(f"  {c.name} {c.type}{null}{note}")
        return ToolOutcome(ok=True, text="\n".join(lines))

    def dry_run(self, query: str) -> ToolOutcome:
        """Parse + bind the query (tables/columns/types) via EXPLAIN without running it."""
        try:
            validate_select(query)
            self.con.execute(f"EXPLAIN {query}")
        except SqlRejected as exc:
            return ToolOutcome(ok=False, text=str(exc), error=str(exc), is_sql=True)
        except duckdb.Error as exc:
            return ToolOutcome(ok=False, text=f"ERROR: {_err(exc)}", error=_err(exc), is_sql=True)
        return ToolOutcome(
            ok=True, text="OK: query is valid. Call run_sql to execute it.", is_sql=True
        )

    def run_sql(self, query: str) -> ToolOutcome:
        try:
            validate_select(query)
            result = db.run_query(self.con, query)
        except SqlRejected as exc:
            return ToolOutcome(ok=False, text=str(exc), error=str(exc), is_sql=True)
        except duckdb.Error as exc:
            return ToolOutcome(ok=False, text=f"ERROR: {_err(exc)}", error=_err(exc), is_sql=True)
        return ToolOutcome(
            ok=True,
            text=_format_rows(result.columns, result.rows),
            rows=result.rows,
            columns=result.columns,
            is_sql=True,
        )


def _fn(name: str, description: str, params: dict[str, str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {k: {"type": "string", "description": v} for k, v in params.items()},
                "required": list(params),
            },
        },
    }


TOOL_SCHEMAS: list[dict] = [
    _fn("list_tables", "List all table names in the database.", {}),
    _fn(
        "get_schema",
        "Show a table's columns, types, primary keys, and foreign-key references.",
        {"table": "Exact table name from list_tables."},
    ),
    _fn(
        "dry_run",
        "Check that a SELECT query is valid (tables, columns, types) without running it.",
        {"query": "A single DuckDB SELECT statement."},
    ),
    _fn(
        "run_sql",
        "Execute a read-only SELECT query and return its rows. Your LAST successful run_sql "
        "is taken as your answer.",
        {"query": "A single DuckDB SELECT statement."},
    ),
]
