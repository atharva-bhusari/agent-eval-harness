"""Async evaluation runner (Phase 4).

For each (task, config) under an asyncio.Semaphore: check cache -> run agent with a
timeout -> score -> classify -> upsert RunRecord. Transient 429/5xx errors are retried
with exponential backoff + jitter (tenacity). A crashed run is recorded as
``agent_error``; it never aborts the sweep.
"""
