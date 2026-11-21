"""Reference agent under test: LangGraph text-to-SQL agent (Phase 2).

Tools: list_tables(), get_schema(table), dry_run(query), run_sql(query) (read-only;
non-SELECT statements are rejected).
Loop: plan -> get schema -> write SQL -> dry_run -> run_sql -> on DB error, feed the
error back and retry up to Config.max_iters (self-correction loop).
Pure: returns an AgentResult and never touches the metrics store.
"""
