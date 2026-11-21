"""DuckDB access layer (Phase 1).

Responsibilities:
- Load the Northwind sample database into ``data/northwind.duckdb``.
- Schema helpers: ``list_tables()``, ``get_schema(table)``.
- ``canonicalize(rows, ordered=False)``: sort rows, round floats to 4 dp, normalize
  None/NaN, cast Decimal -> float, so result sets compare by value, not by shape.
"""
