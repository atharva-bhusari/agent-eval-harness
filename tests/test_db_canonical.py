"""Tests for result canonicalization, the dump converter, and the built Northwind DB."""

from datetime import date
from decimal import Decimal

import pytest

from eval_harness import db
from eval_harness.db import canonicalize, convert_pg_dump, normalize_value, results_match

# --------------------------------------------------------------------------------------
# normalize_value
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (2, 2.0),
        (Decimal("2.50"), 2.5),
        (1 / 3, 0.3333),
        (float("nan"), None),
        (None, None),
        (-0.0, 0.0),
        (date(1996, 7, 4), "1996-07-04"),
        ("Beverages", "Beverages"),
        (True, True),
    ],
)
def test_normalize_value(value, expected):
    assert normalize_value(value) == expected


def test_negative_zero_is_positive_zero():
    assert str(normalize_value(-0.0)) == "0.0"


# --------------------------------------------------------------------------------------
# canonicalize / results_match
# --------------------------------------------------------------------------------------


def test_spec_example_unordered_rows_match():
    # The Phase 1 acceptance example from the spec.
    assert results_match([(2, 1.0), (1, 2.0)], [(1, 2.0), (2, 1.0)])


def test_ordered_flag_makes_row_order_matter():
    assert not results_match([(1,), (2,)], [(2,), (1,)], ordered=True)
    assert results_match([(1,), (2,)], [(1,), (2,)], ordered=True)


def test_duplicates_count_multiset_not_set():
    assert not results_match([(1,), (1,)], [(1,)])
    assert not results_match([(1,), (1,), (2,)], [(1,), (2,), (2,)])


def test_numeric_types_are_equivalent():
    assert results_match([(2,)], [(2.0,)])
    assert results_match([(Decimal("19.99"),)], [(19.99,)])


def test_float_rounding_tolerance():
    assert results_match([(0.1 + 0.2,)], [(0.3,)])
    assert not results_match([(0.3,)], [(0.31,)])


def test_null_and_nan_are_equivalent():
    assert results_match([(None,)], [(float("nan"),)])


def test_yaml_lists_match_db_tuples():
    # expected_result comes from YAML as lists; DuckDB returns tuples.
    assert results_match([["Beverages", 12]], [("Beverages", 12)])


def test_dates_match_iso_strings():
    assert results_match([("1996-07-04",)], [(date(1996, 7, 4),)])


def test_column_order_ignored_via_consistent_permutation():
    assert results_match([("Chai", 18.0), ("Chang", 19.0)], [(18.0, "Chai"), (19.0, "Chang")])


def test_column_permutation_must_be_consistent_across_rows():
    # Swapping values in only one row is a different result, not a column reorder.
    assert not results_match([(1, 2), (3, 4)], [(2, 1), (3, 4)])


def test_column_order_strict_when_disabled():
    assert not results_match([("a", 1)], [(1, "a")], ignore_column_order=False)


def test_shape_mismatches_fail():
    assert not results_match([(1,)], [(1,), (2,)])  # row count
    assert not results_match([(1, 2)], [(1,)])  # column count
    assert not results_match([(1,)], None)  # agent returned nothing


def test_empty_results():
    assert results_match([], [])
    assert not results_match([(1,)], [])


def test_strings_are_exact():
    assert not results_match([("Beverages",)], [("beverages",)])


def test_canonicalize_sorts_mixed_types_with_nulls():
    assert canonicalize([("b",), (None,), ("a",)]) == [(None,), ("a",), ("b",)]
    assert canonicalize([(2,), (1,)], ordered=True) == [(2.0,), (1.0,)]


# --------------------------------------------------------------------------------------
# convert_pg_dump (no network, synthetic dump)
# --------------------------------------------------------------------------------------
MINI_DUMP = """\
SET client_encoding = 'UTF8';
CREATE TABLE parent (
    id smallint NOT NULL,
    price real,
    picture bytea
);
CREATE TABLE child (
    id smallint NOT NULL,
    parent_id smallint
);
INSERT INTO parent VALUES (1, 9.5, '\\x');
INSERT INTO child VALUES (10, 1);
ALTER TABLE ONLY parent
    ADD CONSTRAINT pk_parent PRIMARY KEY (id);
ALTER TABLE ONLY child
    ADD CONSTRAINT pk_child PRIMARY KEY (id);
ALTER TABLE ONLY child
    ADD CONSTRAINT fk_child_parent FOREIGN KEY (parent_id) REFERENCES parent;
"""


def test_convert_pg_dump_loads_into_duckdb():
    import duckdb

    converted = convert_pg_dump(MINI_DUMP)
    assert "SET " not in converted.script
    assert "ALTER TABLE ONLY" not in converted.script

    con = duckdb.connect()
    con.execute(converted.script)
    for stmt in converted.post:
        con.execute(stmt)

    cols = {c.name: c for c in db.get_schema(con, "parent")}
    assert "picture" not in cols  # bytea placeholder dropped
    assert cols["price"].type == "DOUBLE"  # real -> DOUBLE
    assert cols["id"].comment == "primary key"
    child = {c.name: c for c in db.get_schema(con, "child")}
    assert child["parent_id"].comment == "references parent.id"  # FK resolved to PK


def test_convert_pg_dump_rejects_unknown_alter():
    bad = MINI_DUMP + "ALTER TABLE ONLY parent\n    ADD CONSTRAINT u UNIQUE (price);\n"
    with pytest.raises(ValueError, match="unrecognized ALTER"):
        convert_pg_dump(bad)


# --------------------------------------------------------------------------------------
# Integration: the real Northwind DB (skipped until `make db` has run)
# --------------------------------------------------------------------------------------
needs_db = pytest.mark.skipif(
    not db.DEFAULT_DB_PATH.exists(), reason="run `make db` to build data/northwind.duckdb"
)


@pytest.fixture
def con():
    with db.connect() as c:
        yield c


@needs_db
def test_northwind_tables_and_row_counts(con):
    tables = db.list_tables(con)
    for table, n in db.EXPECTED_ROW_COUNTS.items():
        assert table in tables
        assert db.run_query(con, f"SELECT COUNT(*) FROM {table}").rows == [(n,)]


@needs_db
def test_northwind_schema_exposes_join_paths(con):
    cols = {c.name: c for c in db.get_schema(con, "orders")}
    assert cols["customer_id"].comment == "references customers.customer_id"
    assert cols["order_id"].comment == "primary key"


@needs_db
def test_unknown_table_raises(con):
    with pytest.raises(ValueError, match="unknown table"):
        db.get_schema(con, "invoices")


@needs_db
def test_connection_is_read_only(con):
    import duckdb

    with pytest.raises(duckdb.Error):
        con.execute("DELETE FROM orders")


@needs_db
def test_end_to_end_execution_accuracy(con):
    # Two different SQL strings, same rows: both must count as correct.
    a = db.run_query(
        con,
        "SELECT category_name, COUNT(*) FROM categories c "
        "JOIN products p USING (category_id) GROUP BY 1",
    ).rows
    b = db.run_query(
        con,
        "SELECT COUNT(*) AS n, (SELECT category_name FROM categories c "
        "WHERE c.category_id = p.category_id) FROM products p "
        "GROUP BY p.category_id ORDER BY n DESC",
    ).rows
    assert results_match(a, b)
