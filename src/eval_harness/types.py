"""Core contracts shared by every component of the harness.

Everything else (agent, runner, scorers, classifier, store, report) depends on these
types and nothing else, so they are defined first and kept dependency-free.

Scoring is defined against ``Task.expected_result``, never ``Task.expected_sql``:
two different SQL strings that return the same rows are both correct.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# --------------------------------------------------------------------------------------
# Failure taxonomy (spec §5.3). Fixed set: the classifier must only ever emit one of these.
# --------------------------------------------------------------------------------------
FAILURE_CATEGORIES: tuple[str, ...] = (
    "sql_syntax_error",  # DB raised a parse/syntax error on final SQL
    "schema_hallucination",  # referenced a table/column not in the schema
    "wrong_result",  # ran cleanly but result set != expected
    "empty_result",  # returned 0 rows when expected > 0
    "timeout",  # exceeded Config.timeout_s
    "max_iterations",  # hit Config.max_iters without a successful run
    "wrong_tool",  # called a non-existent/disallowed tool, or wrote to the DB
    "agent_error",  # uncaught exception in the agent
    "other",  # none of the above
)

DIFFICULTIES: tuple[str, ...] = ("easy", "medium", "hard")


@dataclass(frozen=True)
class Task:
    """One golden task. Frozen: the golden set must never be mutated mid-run."""

    id: str
    question: str
    expected_result: list[tuple]  # canonical rows; source of truth for scoring
    expected_sql: str | None = None  # reference only, NOT used for scoring
    difficulty: str = "medium"  # easy | medium | hard
    tags: list[str] = field(default_factory=list)
    ordered: bool = False  # True for ranking questions where row order matters

    def __post_init__(self) -> None:
        if self.difficulty not in DIFFICULTIES:
            raise ValueError(f"difficulty must be one of {DIFFICULTIES}, got {self.difficulty!r}")


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    ok: bool
    error: str | None = None


@dataclass
class AgentResult:
    """Everything the agent-under-test reports about a single run. The agent never
    touches the metrics store; it only returns this object."""

    answer_rows: list[tuple] | None  # the result set the agent returned
    final_sql: str | None
    tool_calls: list[ToolCall]
    n_retries: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: int
    error: str | None = None  # set if the agent hard-failed
    stop_reason: str = "answered"  # one of STOP_REASONS; tells the classifier why it ended
    n_nudges: int = 0  # times the agent reminded the model to actually call run_sql
    n_recovered_calls: int = 0  # tool calls parsed from JSON text instead of native calls
    n_llm_calls: int = 0
    n_cached_calls: int = 0  # LLM calls served from the response cache
    # cost_usd / latency_ms are *nominal*: cache hits replay the cost and latency measured
    # when the call really ran, so configs stay comparable on re-runs. spent_usd is what
    # this run actually cost (0 for cache hits).
    spent_usd: float = 0.0
    messages: list[dict] = field(default_factory=list)  # full transcript, for drill-down


STOP_REASONS: tuple[str, ...] = (
    "answered",  # model gave a final reply (answer_rows may still be None/wrong)
    "max_iters",  # hit Config.max_iters model turns without a final reply
    "timeout",  # ran past Config.timeout_s
    "error",  # uncaught exception (LLM/provider failure, bug); see AgentResult.error
)


@runtime_checkable
class Agent(Protocol):
    """Any agent the harness can evaluate. Structural typing: no inheritance required."""

    def run(self, question: str) -> AgentResult: ...


@dataclass
class Config:
    """One evaluation configuration = model x prompt version (+ run limits)."""

    id: str
    model: str  # litellm model id, e.g. "openai/gpt-4o-mini"
    prompt_version: str
    temperature: float = 0.0
    # Max model turns per run. 12, not the spec's 6: with one tool call per turn, a clean
    # v1 run already takes ~6 turns (list_tables, 2x get_schema, dry_run, run_sql, reply),
    # which would leave no budget for self-correction.
    max_iters: int = 12
    timeout_s: int = 60
    # Runs in flight at once. 5 suits hosted APIs; use 1 for a single local GPU (Ollama),
    # where parallel requests just queue up and time out.
    concurrency: int = 5


@dataclass(frozen=True)
class Score:
    success: bool
    detail: str


@dataclass
class RunRecord:
    """One row in the metrics store. Keyed by a deterministic ``run_id`` so writes are
    idempotent upserts."""

    run_id: str  # hash(task.id, config.id, prompt_version)
    task_id: str
    config_id: str
    model: str
    prompt_version: str
    success: bool
    failure_category: str | None
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    n_tool_calls: int
    n_retries: int
    final_sql: str | None
    ts: str  # ISO timestamp
    # --- extensions beyond the spec's columns (diagnostics + drill-down) ---------------
    difficulty: str = "medium"
    stop_reason: str = "answered"
    error: str | None = None
    score_detail: str = ""
    failure_detail: str | None = None  # classifier's one-line reason (failed runs only)
    n_nudges: int = 0
    n_recovered_calls: int = 0
    n_llm_calls: int = 0
    n_cached_calls: int = 0
    spent_usd: float = 0.0
    answer_json: str | None = None  # agent's rows (truncated), JSON
    transcript_json: str | None = None  # full message transcript, JSON

    def __post_init__(self) -> None:
        if self.success and self.failure_category is not None:
            raise ValueError("a successful run cannot have a failure_category")
        if not self.success and self.failure_category not in FAILURE_CATEGORIES:
            raise ValueError(
                f"failed run needs failure_category in {FAILURE_CATEGORIES}, "
                f"got {self.failure_category!r}"
            )


def make_run_id(task_id: str, config_id: str, prompt_version: str) -> str:
    """Deterministic run id: the same (task, config, prompt) always maps to the same row."""
    key = f"{task_id}\x1f{config_id}\x1f{prompt_version}".encode()
    return hashlib.sha256(key).hexdigest()[:16]
