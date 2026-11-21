"""DuckDB access layer: build the Northwind DB, inspect its schema, canonicalize results.

Responsibilities:
- ``build_db()``: download the Northwind Postgres dump (pinned commit + SHA-256), convert
  it to DuckDB SQL, and write ``data/northwind.duckdb`` atomically.
- Schema helpers used by the agent's tools: ``list_tables()``, ``get_schema(table)``.
- ``canonicalize()`` / ``results_match()``: compare result sets by *value*, not by shape.
  This is the foundation of execution-accuracy scoring.

CLI:  ``python -m eval_harness.db build [--force]``
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import math
import os
import re
import urllib.request
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb

from eval_harness.settings import env, repo_root

# --------------------------------------------------------------------------------------
# Paths & source
# --------------------------------------------------------------------------------------
# The dump is downloaded, not vendored: the source repo has no clear license, so we
# don't redistribute it. Pinning the commit + hash keeps builds reproducible.
# These are public, non-secret values: the committed defaults are the reviewed pin, and
# NORTHWIND_SOURCE_URL / NORTHWIND_SOURCE_SHA256 (environment or .env) override them,
# e.g. to use a mirror. Override both together: the hash must match the file at the URL.
_DEFAULT_SOURCE_URL = (
    "https://raw.githubusercontent.com/pthom/northwind_psql/"
    "4aa7967a33b84e693db56d4c23ada410a3ff04a6/northwind.sql"
)
_DEFAULT_SOURCE_SHA256 = "0ee30c01ba282f7194f38bf7f99cd6be0470b7ee5f67d0f7ca41fb058d735e0c"

SOURCE_URL = env("NORTHWIND_SOURCE_URL", _DEFAULT_SOURCE_URL)
SOURCE_SHA256 = env("NORTHWIND_SOURCE_SHA256", _DEFAULT_SOURCE_SHA256).lower()

DATA_DIR = repo_root() / "data"
DEFAULT_DB_PATH = Path(env("EVAL_HARNESS_DB", str(DATA_DIR / "northwind.duckdb")))
RAW_SQL_PATH = DATA_DIR / "raw" / "northwind_psql.sql"

# Expected row counts after build: a sanity check that the load was complete.
EXPECTED_ROW_COUNTS = {
    "categories": 8,
    "customers": 91,
    "employees": 9,
    "employee_territories": 49,
    "order_details": 2155,
    "orders": 830,
    "products": 77,
    "region": 4,
    "shippers": 6,
    "suppliers": 29,
    "territories": 53,
    "us_states": 51,
}


# --------------------------------------------------------------------------------------
# Postgres dump -> DuckDB conversion (pure, unit-testable, no I/O)
# --------------------------------------------------------------------------------------
_SET_RE = re.compile(r"^SET [^;]*;[ \t]*$", re.M)
_CREATE_RE = re.compile(r"CREATE TABLE (\w+) \((.*?)\n\);", re.S)
_ALTER_RE = re.compile(r"ALTER TABLE ONLY [^;]*;", re.S)
_PK_RE = re.compile(r"ALTER TABLE ONLY (\w+)\s+ADD CONSTRAINT \w+ PRIMARY KEY \(([^)]*)\);")
_FK_RE = re.compile(
    r"ALTER TABLE ONLY (\w+)\s+ADD CONSTRAINT \w+ FOREIGN KEY \(([^)]*)\)\s+"
    r"REFERENCES (\w+)(?:\s*\(([^)]*)\))?;"
)


def _cols(s: str) -> list[str]:
    return [c.strip() for c in s.split(",") if c.strip()]


@dataclass
class ConvertedDump:
    script: str  # DROP / CREATE / INSERT statements, DuckDB-compatible
    post: list[str] = field(default_factory=list)  # PKs, column drops, key comments


def convert_pg_dump(sql: str) -> ConvertedDump:
    """Convert the Northwind pg_dump into DuckDB SQL.

    - ``SET ...`` session settings: removed.
    - ``real`` (float32) -> ``DOUBLE``, so money sums don't pick up float32 noise.
    - ``bytea`` image placeholders -> loaded as VARCHAR, then the columns are dropped.
    - ``PRIMARY KEY`` constraints -> ``ALTER TABLE ... ADD PRIMARY KEY`` (enforced).
    - ``FOREIGN KEY`` constraints -> column comments (``references t.col``). Not enforced,
      because the dump inserts child rows before parents; the comments still let the
      agent's ``get_schema`` tool show join paths.
    Any ALTER statement it doesn't recognize raises, so schema changes fail loudly.
    """
    pks: dict[str, list[str]] = {t: _cols(c) for t, c in _PK_RE.findall(sql)}
    fks = _FK_RE.findall(sql)
    n_alters = len(_ALTER_RE.findall(sql))
    if n_alters != len(pks) + len(fks):
        raise ValueError(
            f"unrecognized ALTER statements: {n_alters} total, "
            f"{len(pks)} PK + {len(fks)} FK understood"
        )

    drop_cols: list[tuple[str, str]] = []

    def _fix_create(m: re.Match[str]) -> str:
        table, body = m.group(1), m.group(2)
        for col in re.findall(r"^\s*(\w+)\s+bytea\b", body, re.M):
            drop_cols.append((table, col))
        body = re.sub(r"\breal\b", "DOUBLE", body)
        body = re.sub(r"\bbytea\b", "VARCHAR", body)
        return f"CREATE TABLE {table} ({body}\n);"

    script = _SET_RE.sub("", sql)
    script = _CREATE_RE.sub(_fix_create, script)
    script = _ALTER_RE.sub("", script)

    comments: dict[tuple[str, str], list[str]] = {}
    for table, cols in pks.items():
        for col in cols:
            comments.setdefault((table, col), []).append("primary key")
    for table, cols, ref_table, ref_cols in fks:
        src, dst = _cols(cols), _cols(ref_cols) or pks.get(ref_table, [])
        if len(src) != len(dst):
            raise ValueError(f"cannot resolve FK {table}({cols}) -> {ref_table}")
        for s, d in zip(src, dst, strict=True):
            comments.setdefault((table, s), []).append(f"references {ref_table}.{d}")

    post = [f"ALTER TABLE {t} ADD PRIMARY KEY ({', '.join(c)});" for t, c in pks.items()]
    post += [f"ALTER TABLE {t} DROP COLUMN {c};" for t, c in drop_cols]
    for (table, col), notes in sorted(comments.items()):
        text = "; ".join(notes).replace("'", "''")
        post.append(f"COMMENT ON COLUMN {table}.{col} IS '{text}';")
    return ConvertedDump(script=script, post=post)


# --------------------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_source(dest: Path = RAW_SQL_PATH, url: str = SOURCE_URL) -> Path:
    """Download the pinned dump (skipped if a verified copy already exists)."""
    if dest.exists() and _sha256(dest) == SOURCE_SHA256:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 (pinned https URL)
        data = resp.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SOURCE_SHA256:
        raise RuntimeError(f"checksum mismatch for {url}: got {digest}")
    dest.write_bytes(data)
    return dest


def build_db(
    db_path: Path = DEFAULT_DB_PATH, source: Path | None = None, force: bool = False
) -> Path:
    """Build the DuckDB file. Written to a temp file and renamed, so a failed build never
    leaves a half-loaded database behind."""
    db_path = Path(db_path)
    if db_path.exists() and not force:
        return db_path
    source = source or download_source()
    converted = convert_pg_dump(source.read_text(encoding="utf-8"))

    tmp = db_path.with_suffix(db_path.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(tmp))
    try:
        con.execute(converted.script)
        for stmt in converted.post:
            con.execute(stmt)
        for table, expected in EXPECTED_ROW_COUNTS.items():
            (n,) = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            if n != expected:
                raise RuntimeError(f"{table}: expected {expected} rows, loaded {n}")
    finally:
        con.close()
    os.replace(tmp, db_path)
    return db_path


# --------------------------------------------------------------------------------------
# Connections & schema helpers
# --------------------------------------------------------------------------------------
def connect(db_path: Path | str = DEFAULT_DB_PATH, read_only: bool = True):
    """Open the DB. Read-only by default: the agent must never be able to write."""
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"{db_path} not found; run `make db` first")
    return duckdb.connect(str(db_path), read_only=read_only)


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    nullable: bool
    comment: str | None = None  # e.g. "primary key" / "references orders.order_id"


@dataclass(frozen=True)
class QueryResult:
    columns: list[str]
    rows: list[tuple]


def list_tables(con) -> list[str]:
    rows = con.execute(
        "SELECT table_name FROM duckdb_tables() WHERE schema_name = 'main' ORDER BY table_name"
    ).fetchall()
    return [r[0] for r in rows]


def get_schema(con, table: str) -> list[Column]:
    """Columns of ``table``. Raises ValueError for unknown tables, which is the signal the
    classifier uses to detect schema hallucination."""
    if table not in list_tables(con):
        raise ValueError(f"unknown table {table!r}")
    rows = con.execute(
        "SELECT column_name, data_type, is_nullable, comment FROM duckdb_columns() "
        "WHERE schema_name = 'main' AND table_name = ? ORDER BY column_index",
        [table],
    ).fetchall()
    return [Column(n, t, bool(nullable), c or None) for n, t, nullable, c in rows]


def run_query(con, sql: str) -> QueryResult:
    cur = con.execute(sql)
    columns = [d[0] for d in cur.description] if cur.description else []
    return QueryResult(columns=columns, rows=[tuple(r) for r in cur.fetchall()])


# --------------------------------------------------------------------------------------
# Result canonicalization
# --------------------------------------------------------------------------------------
FLOAT_DP = 4
MAX_PERMUTED_COLUMNS = 6  # 6! = 720 permutations; beyond that, require exact column order


def normalize_value(v: Any) -> Any:
    """Map equivalent values to one representation.

    - Decimal / int / float -> float rounded to 4 dp (so ``2 == 2.0 == Decimal('2.00')``)
    - NaN -> None (SQL NULL); ``-0.0`` -> ``0.0``
    - date / datetime / time -> ISO string (YAML and DuckDB disagree on date types)
    - bool stays bool; strings are kept exactly (case and whitespace matter)
    """
    if v is None or isinstance(v, bool | str):
        return v
    if isinstance(v, int | float | Decimal):
        f = float(v)
        if math.isnan(f):
            return None
        return round(f, FLOAT_DP) + 0.0  # + 0.0 turns -0.0 into 0.0
    if isinstance(v, datetime | date | time):
        return v.isoformat()
    if isinstance(v, bytes | bytearray | memoryview):
        return bytes(v).hex()
    return str(v)


def _sort_key(row: tuple) -> tuple:
    # Total order across mixed types: None < bool < number < string < other.
    rank = {type(None): 0, bool: 1, float: 2, str: 3}
    return tuple((rank.get(type(v), 4), "" if v is None else v) for v in row)


def normalize_rows(rows: Iterable[Sequence[Any]]) -> list[tuple]:
    return [tuple(normalize_value(v) for v in row) for row in rows]


def canonicalize(rows: Iterable[Sequence[Any]], ordered: bool = False) -> list[tuple]:
    """Normalize every value; sort rows unless order is meaningful (ranking questions)."""
    out = normalize_rows(rows)
    return out if ordered else sorted(out, key=_sort_key)


def results_match(
    expected: Iterable[Sequence[Any]],
    actual: Iterable[Sequence[Any]] | None,
    ordered: bool = False,
    ignore_column_order: bool = True,
) -> bool:
    """Execution-accuracy comparison of two result sets.

    Rows are compared as a **multiset** (duplicates count) unless ``ordered``. Expected
    results carry no column labels, so with ``ignore_column_order`` we accept the actual
    result if *one* column permutation, applied to every row, makes it match. This lets
    ``SELECT name, total`` match ``SELECT total, name`` without letting values swap
    between columns row by row.
    """
    if actual is None:
        return False
    exp, act = normalize_rows(expected), normalize_rows(actual)
    if len(exp) != len(act):
        return False
    if not exp:
        return True
    width = len(exp[0])
    if any(len(r) != width for r in exp) or any(len(r) != width for r in act):
        return False

    def _same(candidate: list[tuple]) -> bool:
        if ordered:
            return candidate == exp
        return Counter(candidate) == Counter(exp)

    if _same(act):
        return True
    if not ignore_column_order or width > MAX_PERMUTED_COLUMNS:
        return False
    for perm in itertools.permutations(range(width)):
        if _same([tuple(r[i] for i in perm) for r in act]):
            return True
    return False


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval_harness.db")
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="download Northwind and build the DuckDB file")
    b.add_argument("--force", action="store_true", help="rebuild even if the file exists")
    b.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    args = parser.parse_args(argv)

    if args.cmd == "build":
        path = build_db(args.db, force=args.force)
        with connect(path) as con:
            tables = list_tables(con)
        print(f"built {path} ({len(tables)} tables: {', '.join(tables)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
