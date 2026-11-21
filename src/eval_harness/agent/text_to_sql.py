"""Reference agent under test: a LangGraph text-to-SQL agent with a self-correction loop.

Graph:

    START -> model --(tool calls?)--> tools --> model ... --(final reply / limit)--> END

- ``model`` calls the LLM with the transcript and the four tool schemas.
- ``tools`` executes the requested tools (list_tables, get_schema, dry_run, run_sql) and
  appends their outputs, including DB errors, so the model can fix its query and retry.
- The loop ends when the model replies without tool calls ("answered"), when it has used
  ``max_iters`` model turns ("max_iters"), or on an exception ("error").

The answer is the result of the LAST successful ``run_sql``. The agent is pure: it returns
an ``AgentResult`` and never touches the metrics store.

CLI:  python -m eval_harness.agent.text_to_sql "How many orders shipped to France?"
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from eval_harness import db
from eval_harness.agent.prompts import system_prompt
from eval_harness.agent.sql_tools import TOOL_SCHEMAS, SqlTools
from eval_harness.llm import LLM, LiteLLMClient, ToolCallRequest
from eval_harness.types import AgentResult, Config, ToolCall

SQL_TOOLS = {"dry_run", "run_sql"}
TOOL_NAMES = {t["function"]["name"] for t in TOOL_SCHEMAS}

MAX_NUDGES = 1
NUDGE_MESSAGE = (
    "You have not executed a query with the run_sql tool yet, so there is no answer. "
    "Call run_sql now with the single SELECT statement that answers the question. "
    "Do not write SQL or tool calls as plain text."
)


class AgentState(TypedDict):
    messages: list[dict]
    pending: list[ToolCallRequest]
    turns: int
    tool_calls: list[ToolCall]
    answer_rows: list[tuple] | None
    final_sql: str | None
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    stop_reason: str | None
    error: str | None
    n_nudges: int
    n_recovered_calls: int


def count_retries(tool_calls: list[ToolCall]) -> int:
    """Self-corrections: failed SQL attempts (dry_run/run_sql) followed by another attempt."""
    sql = [c for c in tool_calls if c.name in SQL_TOOLS]
    return sum(1 for i, c in enumerate(sql) if not c.ok and i < len(sql) - 1)


def recover_text_tool_calls(content: str | None) -> list[ToolCallRequest]:
    """Find tool calls the model wrote as JSON text instead of real tool calls.

    Small local models (e.g. llama3.1:8b on Ollama) sometimes reply with
    ``{"name": "run_sql", "parameters": {"query": "..."}}`` in the message body. Any JSON
    object whose ``name`` is one of our tools and whose ``parameters``/``arguments`` is an
    object (or a JSON string of one) is turned into a ``ToolCallRequest``.
    """
    if not content or "{" not in content:
        return []
    decoder = json.JSONDecoder()
    found: list[ToolCallRequest] = []
    i = content.find("{")
    while i != -1:
        try:
            obj, end = decoder.raw_decode(content, i)
        except json.JSONDecodeError:
            i = content.find("{", i + 1)
            continue
        if isinstance(obj, dict) and obj.get("name") in TOOL_NAMES:
            args = obj.get("parameters", obj.get("arguments", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = None
            if isinstance(args, dict):
                found.append(
                    ToolCallRequest(
                        id=f"text_call_{len(found)}",
                        name=obj["name"],
                        args=args,
                        raw_args=json.dumps(args),
                    )
                )
        i = content.find("{", end)
    return found


class TextToSQLAgent:
    """Satisfies ``eval_harness.types.Agent``."""

    def __init__(
        self,
        llm: LLM,
        prompt_version: str = "v1",
        max_iters: int = 12,
        db_path: Path | str = db.DEFAULT_DB_PATH,
    ):
        self.llm = llm
        self.max_iters = max_iters
        self.con = db.connect(db_path, read_only=True)
        self.system_prompt = system_prompt(prompt_version, self.con)
        self.graph = self._build_graph()
        self.last_messages: list[dict] = []  # transcript of the latest run, for debugging

    @classmethod
    def from_config(cls, config: Config, llm: LLM | None = None, **kwargs: Any):
        llm = llm or LiteLLMClient(config.model, temperature=config.temperature)
        return cls(llm, prompt_version=config.prompt_version, max_iters=config.max_iters, **kwargs)

    # ---- graph ----------------------------------------------------------------------
    def _build_graph(self):
        g = StateGraph(AgentState)
        g.add_node("model", self._model_node)
        g.add_node("tools", self._tools_node)
        g.add_edge(START, "model")
        g.add_conditional_edges("model", self._route, ["tools", "model", END])
        g.add_edge("tools", "model")
        return g.compile()

    @staticmethod
    def _route(state: AgentState) -> str:
        if state["stop_reason"]:
            return END
        return "tools" if state["pending"] else "model"  # no pending calls = nudged

    def _model_node(self, state: AgentState) -> dict:
        if state["turns"] >= self.max_iters:
            return {"stop_reason": "max_iters"}
        try:
            resp = self.llm(state["messages"], TOOL_SCHEMAS)
        except Exception as exc:  # provider/network failure: record it, keep the trace
            return {"stop_reason": "error", "error": f"{type(exc).__name__}: {exc}"}

        calls = list(resp.tool_calls)
        recovered = 0
        if not calls:  # fix 1: tool calls written as JSON text
            calls = recover_text_tool_calls(resp.content)
            recovered = len(calls)

        assistant: dict[str, Any] = {"role": "assistant", "content": resp.content or ""}
        if calls:
            assistant["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": c.raw_args or "{}"},
                }
                for c in calls
            ]
        messages = [*state["messages"], assistant]

        stop_reason: str | None = None
        nudges = state["n_nudges"]
        if not calls:
            # fix 2: the model is trying to finish without ever running a query.
            if state["answer_rows"] is None and nudges < MAX_NUDGES:
                messages.append({"role": "user", "content": NUDGE_MESSAGE})
                nudges += 1
            else:
                stop_reason = "answered"

        return {
            "messages": messages,
            "pending": calls,
            "turns": state["turns"] + 1,
            "prompt_tokens": state["prompt_tokens"] + resp.prompt_tokens,
            "completion_tokens": state["completion_tokens"] + resp.completion_tokens,
            "cost_usd": state["cost_usd"] + resp.cost_usd,
            "stop_reason": stop_reason,
            "n_nudges": nudges,
            "n_recovered_calls": state["n_recovered_calls"] + recovered,
        }

    def _tools_node(self, state: AgentState) -> dict:
        tools = SqlTools(self._cursor)
        messages = list(state["messages"])
        calls = list(state["tool_calls"])
        answer_rows, final_sql = state["answer_rows"], state["final_sql"]
        for req in state["pending"]:
            out = tools.call(req.name, req.args)
            calls.append(ToolCall(name=req.name, args=req.args or {}, ok=out.ok, error=out.error))
            messages.append(
                {"role": "tool", "tool_call_id": req.id, "name": req.name, "content": out.text}
            )
            if req.name == "run_sql" and out.ok:
                answer_rows, final_sql = out.rows, (req.args or {}).get("query")
        return {
            "messages": messages,
            "pending": [],
            "tool_calls": calls,
            "answer_rows": answer_rows,
            "final_sql": final_sql,
        }

    # ---- Agent protocol -------------------------------------------------------------
    def run(self, question: str) -> AgentResult:
        start = time.perf_counter()
        self._cursor = self.con.cursor()  # per-run cursor: safe when runs use threads
        state: AgentState = {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question},
            ],
            "pending": [],
            "turns": 0,
            "tool_calls": [],
            "answer_rows": None,
            "final_sql": None,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cost_usd": 0.0,
            "stop_reason": None,
            "error": None,
            "n_nudges": 0,
            "n_recovered_calls": 0,
        }
        try:
            final = self.graph.invoke(state, {"recursion_limit": 2 * self.max_iters + 5})
        except Exception as exc:  # bug in graph/tools: still return a result, never raise
            final = {**state, "stop_reason": "error", "error": f"{type(exc).__name__}: {exc}"}
        finally:
            self._cursor.close()

        self.last_messages = final["messages"]
        return AgentResult(
            answer_rows=final["answer_rows"],
            final_sql=final["final_sql"],
            tool_calls=final["tool_calls"],
            n_retries=count_retries(final["tool_calls"]),
            prompt_tokens=final["prompt_tokens"],
            completion_tokens=final["completion_tokens"],
            cost_usd=final["cost_usd"],
            latency_ms=int((time.perf_counter() - start) * 1000),
            error=final["error"],
            stop_reason=final["stop_reason"] or "error",
            n_nudges=final["n_nudges"],
            n_recovered_calls=final["n_recovered_calls"],
        )


# --------------------------------------------------------------------------------------
# CLI: ask one question and print the full trace
# --------------------------------------------------------------------------------------
def _print_trace(messages: list[dict]) -> None:
    for m in messages[1:]:  # skip the system prompt
        if m["role"] == "user":
            print(f"\nUSER: {m['content']}")
        elif m["role"] == "assistant":
            if m.get("content"):
                print(f"\nMODEL: {m['content']}")
            for c in m.get("tool_calls", []):
                print(f"\n-> {c['function']['name']}({c['function']['arguments']})")
        elif m["role"] == "tool":
            print("   " + m["content"].replace("\n", "\n   "))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m eval_harness.agent.text_to_sql")
    p.add_argument("question")
    p.add_argument("--model", default="ollama/llama3.1:8b")
    p.add_argument("--prompt", default="v1", help="prompt version (v1 | v2)")
    p.add_argument("--max-iters", type=int, default=12)
    args = p.parse_args(argv)

    agent = TextToSQLAgent(
        LiteLLMClient(args.model), prompt_version=args.prompt, max_iters=args.max_iters
    )
    result = agent.run(args.question)
    _print_trace(agent.last_messages)
    print("\n" + "=" * 70)
    print(
        f"stop_reason: {result.stop_reason}   retries: {result.n_retries}   "
        f"tool calls: {len(result.tool_calls)}   nudges: {result.n_nudges}   "
        f"recovered text calls: {result.n_recovered_calls}"
    )
    print(f"final_sql:   {result.final_sql}")
    print(f"answer_rows: {result.answer_rows[:10] if result.answer_rows else None}")
    print(
        f"tokens: {result.prompt_tokens}+{result.completion_tokens}   "
        f"cost: ${result.cost_usd:.5f}   latency: {result.latency_ms / 1000:.1f}s"
    )
    if result.error:
        print(f"error: {result.error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
