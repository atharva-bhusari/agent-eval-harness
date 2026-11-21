"""SQLite metrics store (Phase 4).

Tables: ``runs`` (one row per RunRecord, PRIMARY KEY run_id) and ``tool_calls``
(drill-down). All writes are upserts keyed by run_id, so re-runs are idempotent.
"""
