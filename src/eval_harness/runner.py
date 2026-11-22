"""Async evaluation runner.

For every task in a config: run the agent (in a worker thread, bounded by a semaphore and
a timeout) -> score -> classify -> upsert a ``RunRecord``. Each run is written as soon as
it finishes, so an interrupted eval keeps its completed runs and ``--resume`` continues.

Reliability rules (spec §1, §5.5):
- A crashed or hung run becomes a failed record (``agent_error`` / ``timeout``); it never
  aborts the eval.
- Timeouts are enforced *inside* the agent (deadline checked each turn and passed to the
  LLM call). A hard ``wait_for`` backstop (timeout + grace) catches anything that ignores it.
- Transient provider errors (429/5xx) are retried with backoff in ``llm.py``.

CLI:  python -m eval_harness.runner --config configs/ollama_llama.yaml [--tasks e01,m03]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from eval_harness import golden
from eval_harness.classify import explain
from eval_harness.config import load_config
from eval_harness.llm import is_local
from eval_harness.scorers import execution_accuracy
from eval_harness.store import DEFAULT_STORE_PATH, Store
from eval_harness.types import Agent, AgentResult, Config, RunRecord, Task, make_run_id

HARD_TIMEOUT_GRACE_S = 30
MAX_ANSWER_ROWS_STORED = 50


# --------------------------------------------------------------------------------------
# One run -> RunRecord
# --------------------------------------------------------------------------------------
def failed_result(stop_reason: str, error: str, latency_ms: int) -> AgentResult:
    return AgentResult(
        answer_rows=None,
        final_sql=None,
        tool_calls=[],
        n_retries=0,
        prompt_tokens=0,
        completion_tokens=0,
        cost_usd=0.0,
        latency_ms=latency_ms,
        error=error,
        stop_reason=stop_reason,
    )


def build_record(task: Task, config: Config, result: AgentResult) -> RunRecord:
    score = execution_accuracy(task, result)
    verdict = None if score.success else explain(task, result)
    rows = result.answer_rows
    return RunRecord(
        run_id=make_run_id(task.id, config.id, config.prompt_version),
        task_id=task.id,
        config_id=config.id,
        model=config.model,
        prompt_version=config.prompt_version,
        success=score.success,
        failure_category=verdict.category if verdict else None,
        latency_ms=result.latency_ms,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        cost_usd=result.cost_usd,
        n_tool_calls=len(result.tool_calls),
        n_retries=result.n_retries,
        final_sql=result.final_sql,
        ts=datetime.now(UTC).isoformat(timespec="seconds"),
        difficulty=task.difficulty,
        stop_reason=result.stop_reason,
        error=result.error,
        score_detail=score.detail,
        failure_detail=verdict.reason if verdict else None,
        n_nudges=result.n_nudges,
        n_recovered_calls=result.n_recovered_calls,
        n_llm_calls=result.n_llm_calls,
        n_cached_calls=result.n_cached_calls,
        spent_usd=result.spent_usd,
        answer_json=None
        if rows is None
        else json.dumps([list(r) for r in rows[:MAX_ANSWER_ROWS_STORED]], default=str),
        transcript_json=json.dumps(result.messages, default=str) if result.messages else None,
    )


async def run_one(agent: Agent, task: Task, config: Config) -> AgentResult:
    start = time.perf_counter()
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(agent.run, task.question),
            timeout=config.timeout_s + HARD_TIMEOUT_GRACE_S,
        )
    except TimeoutError:
        ms = int((time.perf_counter() - start) * 1000)
        return failed_result("timeout", f"hard timeout after {ms / 1000:.0f}s", ms)
    except Exception as exc:  # the agent contract says run() never raises; belt and braces
        ms = int((time.perf_counter() - start) * 1000)
        return failed_result("error", f"{type(exc).__name__}: {exc}", ms)


# --------------------------------------------------------------------------------------
# A whole eval
# --------------------------------------------------------------------------------------
@dataclass
class EvalSummary:
    config_id: str
    records: list[RunRecord] = field(default_factory=list)
    skipped: int = 0
    wall_s: float = 0.0

    @property
    def n(self) -> int:
        return len(self.records)

    @property
    def accuracy(self) -> float:
        return sum(r.success for r in self.records) / self.n if self.n else 0.0

    def render(self) -> str:
        if not self.records:
            return f"{self.config_id}: nothing run ({self.skipped} skipped)"
        rs = self.records
        llm_calls = sum(r.n_llm_calls for r in rs)
        cached = sum(r.n_cached_calls for r in rs)
        fails = Counter(r.failure_category for r in rs if not r.success)
        lines = [
            f"config:        {self.config_id}",
            f"accuracy:      {sum(r.success for r in rs)}/{self.n} = {self.accuracy:.1%}",
            f"mean latency:  {sum(r.latency_ms for r in rs) / self.n / 1000:.1f}s",
            f"mean retries:  {sum(r.n_retries for r in rs) / self.n:.2f}",
            f"cost (nominal): ${sum(r.cost_usd for r in rs):.4f}   "
            f"spent this run: ${sum(r.spent_usd for r in rs):.4f}",
            f"cache:         {cached}/{llm_calls} LLM calls served from cache",
            f"wall time:     {self.wall_s:.0f}s"
            + (f"   skipped: {self.skipped}" if self.skipped else ""),
        ]
        if fails:
            lines.append("failures:      " + ", ".join(f"{k}={v}" for k, v in fails.most_common()))
        return "\n".join(lines)


def _progress_line(i: int, n: int, r: RunRecord) -> str:
    mark = "PASS" if r.success else "FAIL"
    cache = " (cached)" if r.n_llm_calls and r.n_cached_calls == r.n_llm_calls else ""
    why = "" if r.success else f"  {r.failure_category}: {r.score_detail}"
    return f"[{i:>2}/{n}] {mark} {r.task_id:<4} {r.latency_ms / 1000:6.1f}s{cache}{why}"


async def run_eval(
    config: Config,
    tasks: Sequence[Task],
    store: Store,
    agent: Agent | None = None,
    concurrency: int | None = None,
    resume: bool = False,
    warm_up: bool = True,
    on_record: Callable[[int, int, RunRecord], None] | None = None,
) -> EvalSummary:
    if agent is None:
        from eval_harness.agent.text_to_sql import TextToSQLAgent

        agent = TextToSQLAgent.from_config(config)

    summary = EvalSummary(config.id)
    if resume:
        done = store.run_ids(config.id)
        todo = [t for t in tasks if make_run_id(t.id, config.id, config.prompt_version) not in done]
        summary.skipped = len(tasks) - len(todo)
        tasks = todo

    llm = getattr(agent, "llm", None)
    if warm_up and is_local(config.model) and hasattr(llm, "warm_up"):
        await asyncio.to_thread(llm.warm_up)  # load the local model before timing anything

    start = time.perf_counter()
    sem = asyncio.Semaphore(concurrency or config.concurrency)
    total, finished = len(tasks), 0

    async def _task(task: Task) -> None:
        nonlocal finished
        async with sem:
            result = await run_one(agent, task, config)
        record = build_record(task, config, result)
        store.upsert(record, result.tool_calls)  # written immediately: crash-safe
        summary.records.append(record)
        finished += 1
        if on_record:
            on_record(finished, total, record)

    await asyncio.gather(*(_task(t) for t in tasks))
    summary.wall_s = time.perf_counter() - start
    return summary


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def select_tasks(tasks: list[Task], ids: str | None, limit: int | None) -> list[Task]:
    if ids:
        wanted = [s.strip() for s in ids.split(",") if s.strip()]
        by_id = {t.id: t for t in tasks}
        unknown = [w for w in wanted if w not in by_id]
        if unknown:
            raise SystemExit(f"unknown task ids: {unknown}")
        tasks = [by_id[w] for w in wanted]
    return tasks[:limit] if limit else tasks


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m eval_harness.runner")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--tasks", help="comma-separated task ids (default: all)")
    p.add_argument("--limit", type=int, help="run only the first N tasks")
    p.add_argument("--resume", action="store_true", help="skip tasks already in the store")
    p.add_argument("--concurrency", type=int, help="override the config's concurrency")
    p.add_argument("--store", type=Path, default=DEFAULT_STORE_PATH)
    p.add_argument("--no-warmup", action="store_true")
    args = p.parse_args(argv)

    config = load_config(args.config)
    tasks = select_tasks(golden.load_tasks(), args.tasks, args.limit)
    print(
        f"eval {config.id}: model={config.model} prompt={config.prompt_version} "
        f"tasks={len(tasks)} concurrency={args.concurrency or config.concurrency} "
        f"timeout={config.timeout_s}s",
        flush=True,
    )
    with Store(args.store) as store:
        summary = asyncio.run(
            run_eval(
                config,
                tasks,
                store,
                concurrency=args.concurrency,
                resume=args.resume,
                warm_up=not args.no_warmup,
                on_record=lambda i, n, r: print(_progress_line(i, n, r), flush=True),
            )
        )
        print("\n" + summary.render())
        print(f"store:         {args.store} ({store.count(config.id)} rows for {config.id})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
