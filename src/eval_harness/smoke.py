"""5-task smoke eval: the whole pipeline end to end, with a scripted LLM (no key, no cost).

Real agent (LangGraph loop, tools, read-only guard), real runner, scorer, classifier,
SQLite store and Markdown report. Only the model is fake. Each task gets one scripted
behaviour so every important path runs, and the expected outcome of each is asserted:

    e01  answers with the reference SQL                -> pass
    e05  answers with the reference SQL                -> pass
    m03  bad column, reads the DB error, then fixes it -> pass, 1 retry (self-correction)
    m08  runs a query that returns the wrong rows      -> wrong_result
    h05  queries a column that doesn't exist, gives up -> schema_hallucination

Exit code 0 only if every outcome matches. This is what CI runs (`make smoke`), so a
regression anywhere in the pipeline fails the build without a model or an API key.

CLI:  python -m eval_harness.smoke [--out DIR]   (default: a temporary directory)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

from eval_harness import golden
from eval_harness.agent.text_to_sql import TextToSQLAgent
from eval_harness.llm import LLMResponse, ToolCallRequest
from eval_harness.report import write_report
from eval_harness.runner import _progress_line, run_eval
from eval_harness.store import Store
from eval_harness.types import Config

# task id -> (behaviour, expected (success, failure_category, min retries))
SCRIPT: dict[str, tuple[str, tuple[bool, str | None, int]]] = {
    "e01": ("correct", (True, None, 0)),
    "e05": ("correct", (True, None, 0)),
    "m03": ("self_correct", (True, None, 1)),
    "m08": ("wrong", (False, "wrong_result", 0)),
    "h05": ("hallucinate", (False, "schema_hallucination", 0)),
}
BAD_COLUMN_SQL = "SELECT shipper_id FROM orders"  # orders has ship_via, not shipper_id
CONFIG = Config(id="smoke", model="mock/scripted", prompt_version="v1", concurrency=5)


def _sql_call(sql: str) -> LLMResponse:
    args = {"query": sql}
    call = ToolCallRequest(id="call_1", name="run_sql", args=args, raw_args=json.dumps(args))
    return LLMResponse(content=None, tool_calls=[call], prompt_tokens=120, completion_tokens=20)


def _reply(text: str) -> LLMResponse:
    return LLMResponse(content=text, prompt_tokens=150, completion_tokens=10)


class ScriptedLLM:
    """Decides each turn from the transcript alone (stateless, so safe under concurrency)."""

    def __init__(self, tasks):
        self.by_question = {t.question: t for t in tasks}

    def __call__(self, messages, tools, timeout=None):
        question = next(m["content"] for m in messages if m["role"] == "user")
        task = self.by_question[question]
        mode = SCRIPT[task.id][0]
        n_tool = sum(m["role"] == "tool" for m in messages)
        if mode == "correct":
            return _sql_call(task.expected_sql) if n_tool == 0 else _reply("Done.")
        if mode == "self_correct":
            if n_tool == 0:
                return _sql_call(BAD_COLUMN_SQL)
            return _sql_call(task.expected_sql) if n_tool == 1 else _reply("Done.")
        if mode == "wrong":
            return _sql_call("SELECT 1 AS answer") if n_tool == 0 else _reply("Done.")
        if mode == "hallucinate":
            return _sql_call(BAD_COLUMN_SQL) if n_tool == 0 else _reply("I can't answer that.")
        raise ValueError(f"unknown behaviour {mode!r}")


def check(records) -> list[str]:
    """Mismatches between what happened and what the script expects (empty = all good)."""
    problems = []
    by_id = {r.task_id: r for r in records}
    for tid, (_, (success, category, min_retries)) in SCRIPT.items():
        r = by_id.get(tid)
        if r is None:
            problems.append(f"{tid}: no run recorded")
            continue
        got = (r.success, r.failure_category)
        if got != (success, category):
            problems.append(f"{tid}: expected {(success, category)}, got {got} ({r.score_detail})")
        if r.n_retries < min_retries:
            problems.append(f"{tid}: expected >= {min_retries} retries, got {r.n_retries}")
    return problems


def run_smoke(out_dir: Path) -> tuple[list[str], Path]:
    tasks = [t for t in golden.load_tasks() if t.id in SCRIPT]
    missing = set(SCRIPT) - {t.id for t in tasks}
    if missing:
        return [f"golden set has no task {sorted(missing)}"], out_dir
    agent = TextToSQLAgent(ScriptedLLM(tasks), prompt_version=CONFIG.prompt_version)
    store_path = out_dir / "smoke.sqlite"
    with Store(store_path) as store:
        summary = asyncio.run(
            run_eval(
                CONFIG,
                tasks,
                store,
                agent=agent,
                warm_up=False,
                on_record=lambda i, n, r: print(_progress_line(i, n, r), flush=True),
            )
        )
        stored = store.count(CONFIG.id)
    print("\n" + summary.render())
    problems = check(summary.records)
    if stored != len(SCRIPT):
        problems.append(f"store has {stored} rows, expected {len(SCRIPT)}")
    report = write_report(
        store_path, out_dir / "REPORT.md", n_tasks=len(SCRIPT), title="Smoke eval"
    )
    if "## Head-to-head" in report.read_text(encoding="utf-8"):  # one config: no pairs
        problems.append("report rendered a head-to-head section for a single config")
    return problems, report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m eval_harness.smoke")
    p.add_argument("--out", type=Path, help="keep the store + report here (default: temp dir)")
    args = p.parse_args(argv)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        problems, report = run_smoke(args.out)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            problems, report = run_smoke(Path(tmp))
            report = Path("(temporary)") / report.name
    if problems:
        print("\nSMOKE FAILED:", *problems, sep="\n  - ", file=sys.stderr)
        return 1
    print(f"\nsmoke OK: all {len(SCRIPT)} scripted outcomes matched · report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
