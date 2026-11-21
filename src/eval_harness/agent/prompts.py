"""Versioned system prompts. ``Config.prompt_version`` selects one.

Phase 7 A/B hypothesis:
- v1: the agent discovers the schema through tools (list_tables / get_schema).
- v2: a compact schema is put in the prompt up front, plus explicit query-writing rules.
  Expected: fewer tool calls and fewer schema hallucinations, at the cost of a longer prompt.

Never edit a prompt in place once results exist for it: add a new version instead, so
stored results stay attributable to the exact prompt that produced them.
"""

from __future__ import annotations

from eval_harness import db

_RULES = """\
Rules:
- The database is DuckDB (PostgreSQL-like SQL). Only single SELECT statements are allowed.
- Use only tables and columns that exist. Never guess a name.
- Always check a new query with dry_run before run_sql.
- If a tool returns an ERROR, read it, fix the query, and try again.
- Your LAST successful run_sql is taken as your final answer, so make it return exactly the
  rows and columns the question asks for: no extra columns, no explanatory rows.
- When the result is correct, reply with one short sentence and do not call any more tools."""

V1 = f"""\
You are a data analyst answering questions about a company's database by writing SQL.

Workflow:
1. Call list_tables, then get_schema for each table you need.
2. Write a query, check it with dry_run, then execute it with run_sql.
3. On any error, fix the query and retry.

{_RULES}"""

_V2_TEMPLATE = """\
You are a careful data analyst answering questions about the Northwind trading company's
database (customers, orders, products, suppliers, employees) by writing DuckDB SQL.

Schema (table: columns; "pk" = primary key, "-> t.c" = foreign key):
{schema}

Query-writing notes:
- Line-item revenue = unit_price * quantity * (1 - discount), from order_details.
- Join through foreign keys exactly as listed; do not invent join columns.
- For "top N" questions use ORDER BY ... LIMIT N. Round money only if asked.

{rules}"""


def compact_schema(con) -> str:
    """One line per table, e.g. ``orders: order_id pk, customer_id -> customers.customer_id``."""
    lines = []
    for table in db.list_tables(con):
        parts = []
        for c in db.get_schema(con, table):
            notes = (c.comment or "").split("; ")
            tag = " pk" if "primary key" in notes else ""
            refs = [n.removeprefix("references ") for n in notes if n.startswith("references")]
            parts.append(f"{c.name}{tag}" + "".join(f" -> {r}" for r in refs))
        lines.append(f"{table}: {', '.join(parts)}")
    return "\n".join(lines)


PROMPT_VERSIONS = ("v1", "v2")


def system_prompt(version: str, con) -> str:
    if version == "v1":
        return V1
    if version == "v2":
        return _V2_TEMPLATE.format(schema=compact_schema(con), rules=_RULES)
    raise ValueError(f"unknown prompt_version {version!r}; known: {PROMPT_VERSIONS}")
