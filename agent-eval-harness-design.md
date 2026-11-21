# Agent Eval Harness — System Design & Build Spec

> Hand this file to a Claude Code agent as the build spec. It is ordered in phases;
> each phase has **deliverables** and **acceptance criteria** the agent can verify
> before moving on. Build strictly in order. Do not skip acceptance checks.

---

## 0. What we are building (one paragraph)

A reusable **evaluation harness for LLM agents**. It runs an agent-under-test against a
fixed **golden task set**, captures a full trace of every run (tool calls, tokens, latency,
cost), scores each answer on **execution accuracy** (did it produce the correct result, not
the correct wording), classifies every failure into a **taxonomy**, and renders an **A/B
comparison across prompt versions and models** as both a Streamlit dashboard and a
Markdown table for the README. The reference agent-under-test is a **text-to-SQL data
agent** over a local DuckDB database, but the agent sits behind an interface so it is
swappable.

The portfolio headline this unlocks: *"Built an eval harness that scores a text-to-SQL
agent on execution accuracy across 50 golden tasks and 4 configs, with a failure taxonomy
and an A/B model comparison."*

---

## 1. Requirements

### Functional
- F1. Define golden tasks as data (`question`, `expected_result`, metadata) — target **50 tasks**.
- F2. Run a pluggable agent against all tasks for one or more **configs** (model × prompt version).
- F3. Capture per-run trace: final answer, generated SQL, tool calls, retries, tokens, latency, cost.
- F4. Score each run with **execution accuracy** (result-set match, order-insensitive).
- F5. Classify each failure into a fixed taxonomy.
- F6. Persist all runs durably; re-runs are idempotent and cacheable.
- F7. Aggregate metrics and emit (a) a Markdown comparison table, (b) a Streamlit dashboard.
- F8. Run an A/B sweep across ≥2 prompts × ≥2 models and compare.

### Non-functional (realistic laptop targets — state these as assumptions)
- Scale: 50 tasks × 4 configs = **200 runs** per full eval; full sweep completes in ~10–30 min.
- Cost: default models are cheap (gpt-4o-mini / Claude Haiku ≈ pennies) or **free** (local Ollama).
- Reproducibility: `temperature=0`; response cache keyed by `hash(task, config, prompt)`.
- Reliability: exponential backoff on HTTP 429/5xx; per-run timeout; a crashed run is recorded as a failure, never aborts the sweep.
- Concurrency: async with a semaphore (default 5 concurrent runs).
- Everything free/OSS and runnable offline except the hosted-model API calls (which have a local Ollama fallback).

### Constraints
- Solo build, ~2 weekends, Python 3.11+.
- Must be a public GitHub repo with a clean README, tests, and CI.

---

## 2. High-Level Design

```
                 ┌──────────────────────────────────────────────┐
                 │                  CONFIG (yaml)                 │
                 │   model, prompt_version, temperature, max_iter │
                 └───────────────┬──────────────────────────────┘
                                 │
   tasks.yaml                    ▼
  (golden set) ───────►  ┌───────────────┐   AgentResult   ┌──────────────┐
                         │    RUNNER      │ ──────────────► │   SCORERS    │
  ┌───────────────┐      │ (async, retry, │                 │ exec-accuracy│
  │ AGENT UNDER    │◄────►│  timeout,     │                 │ (+LLM judge) │
  │ TEST (iface)   │ tools│  cache)       │                 └──────┬───────┘
  │ text-to-SQL    │      └───────┬───────┘                        │
  │ LangGraph      │              │ trace                          ▼
  └───────┬────────┘              │                         ┌──────────────┐
          │ run_sql               ▼                         │  FAILURE     │
          ▼                ┌──────────────┐                 │  CLASSIFIER  │
   ┌─────────────┐         │  METRICS     │◄────────────────┤  (taxonomy)  │
   │  DuckDB     │         │  STORE       │                 └──────────────┘
   │ (Northwind) │         │  (SQLite)    │
   └─────────────┘         └──────┬───────┘
                                  │
                   ┌──────────────┴───────────────┐
                   ▼                              ▼
          ┌─────────────────┐            ┌──────────────────┐
          │ REPORT (md table)│            │ DASHBOARD         │
          │ for README       │            │ (Streamlit)       │
          └─────────────────┘            └──────────────────┘
                   ▲
                   │
            ┌──────┴───────┐
            │ CI smoke eval │  (GitHub Actions: 5 tasks on push)
            └──────────────┘
```

**Data flow:** config + tasks → runner invokes agent (agent queries DuckDB via tools) →
AgentResult → scorer marks success/failure → classifier labels failures → everything
written to SQLite → report + dashboard read SQLite.

---

## 3. Repository layout

```
agent-eval-harness/
├── README.md                  # final: what it is, how to run, results table + screenshots
├── pyproject.toml             # deps via uv/pip; ruff + pytest config
├── Makefile                   # setup, db, eval, report, dashboard, test, smoke
├── .github/workflows/ci.yml   # lint + unit tests + 5-task smoke eval
├── configs/
│   ├── baseline.yaml
│   ├── v2_prompt.yaml
│   ├── haiku.yaml
│   └── gpt4o_mini.yaml
├── data/
│   ├── northwind.duckdb       # built by `make db` (gitignored); dump downloaded from a pinned commit + SHA-256
│   └── tasks.yaml             # 50 golden tasks
├── src/eval_harness/
│   ├── __init__.py
│   ├── types.py               # Task, AgentResult, RunRecord, Config dataclasses + Protocols
│   ├── db.py                  # DuckDB connection + schema helpers + result canonicalization
│   ├── llm.py                 # thin multi-provider client (litellm) + response cache
│   ├── agent/
│   │   ├── base.py            # Agent Protocol
│   │   └── text_to_sql.py     # reference LangGraph agent + tools
│   ├── runner.py              # async orchestration, retry, timeout, idempotent writes
│   ├── scorers.py             # execution-accuracy scorer (+ optional llm_judge)
│   ├── classify.py            # failure taxonomy classifier
│   ├── store.py               # SQLite persistence (runs, task_results)
│   ├── report.py              # aggregate → markdown table
│   └── dashboard.py           # Streamlit app
└── tests/
    ├── test_scorers.py
    ├── test_classify.py
    ├── test_db_canonical.py
    └── test_runner_smoke.py
```

---

## 4. Core contracts (define these first, in `types.py`)

```python
from dataclasses import dataclass, field
from typing import Protocol, Any

@dataclass(frozen=True)
class Task:
    id: str
    question: str
    expected_result: list[tuple]      # canonical rows; source of truth for scoring
    expected_sql: str | None = None   # reference only, NOT used for scoring
    difficulty: str = "medium"        # easy | medium | hard
    tags: list[str] = field(default_factory=list)

@dataclass
class ToolCall:
    name: str
    args: dict
    ok: bool
    error: str | None = None

@dataclass
class AgentResult:
    answer_rows: list[tuple] | None    # the result set the agent returned
    final_sql: str | None
    tool_calls: list[ToolCall]
    n_retries: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: int
    error: str | None = None           # set if the agent hard-failed

class Agent(Protocol):
    def run(self, question: str) -> AgentResult: ...

@dataclass
class Config:
    id: str
    model: str
    prompt_version: str
    temperature: float = 0.0
    max_iters: int = 6
    timeout_s: int = 60

@dataclass
class RunRecord:
    run_id: str            # hash(task.id, config.id, prompt_version)
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
    ts: str                # ISO timestamp
```

**Scoring is defined against `expected_result`, never `expected_sql`.** Two different SQL
strings that return the same rows are both correct.

---

## 5. Deep dives (design decisions the agent must implement)

### 5.1 Reference agent (`agent/text_to_sql.py`)
- Build with **LangGraph** (it is the resume keyword and gives a clean graph + checkpointing).
- Tools: `list_tables()`, `get_schema(table)`, `run_sql(query)` (read-only; reject non-SELECT),
  `dry_run(query)` (EXPLAIN / parse-check before executing).
- Control loop: plan → (get schema) → write SQL → dry_run → run_sql → on error, feed the DB
  error back and retry up to `max_iters`. This **self-correction loop** is a core feature.
- Keep the agent pure: it returns an `AgentResult`; it does **not** touch the metrics store.

### 5.2 Execution-accuracy scorer (`scorers.py`)
- Canonicalize both expected and actual result sets before comparing (in `db.py`):
  sort rows, sort columns if unlabeled, round floats to 4 dp, normalize `None`/NaN, cast
  decimals to float. Comparison is **set/multiset equality**, order-insensitive by default
  (a `ordered=True` flag per task for ranking questions).
- Return `Score{success: bool, detail: str}`.
- Optional second scorer `llm_judge` (off by default) for answer-explanation quality —
  clearly marked as subjective and cost-bearing.

### 5.3 Failure taxonomy (`classify.py`)
Fixed categories (rule-based first; LLM fallback only for `other`):
| category | detected by |
|---|---|
| `sql_syntax_error` | DB raised a parse/syntax error on final SQL |
| `schema_hallucination` | referenced a table/column not in `list_tables`/`get_schema` |
| `wrong_result` | ran cleanly but result set ≠ expected |
| `empty_result` | returned 0 rows when expected > 0 |
| `timeout` | exceeded `timeout_s` |
| `max_iterations` | hit `max_iters` without a successful run |
| `wrong_tool` | called a non-existent/disallowed tool, or wrote to the DB |
| `agent_error` | uncaught exception in the agent |
| `other` | none of the above (LLM may label a sub-reason) |

### 5.4 Metrics store (`store.py`, SQLite)
```sql
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, task_id TEXT, config_id TEXT, model TEXT,
  prompt_version TEXT, success INTEGER, failure_category TEXT,
  latency_ms INTEGER, prompt_tokens INTEGER, completion_tokens INTEGER,
  cost_usd REAL, n_tool_calls INTEGER, n_retries INTEGER,
  final_sql TEXT, ts TEXT
);
CREATE TABLE IF NOT EXISTS tool_calls (   -- optional, for drill-down
  run_id TEXT, seq INTEGER, name TEXT, ok INTEGER, error TEXT
);
```
Writes are **upserts keyed by `run_id`** so re-running a task/config overwrites cleanly (F6).

### 5.5 Runner (`runner.py`)
- `async` over `tasks × configs` with an `asyncio.Semaphore(concurrency)`.
- Per run: check response cache → invoke agent with timeout → score → classify → upsert.
- Backoff: retry transient HTTP 429/5xx with exponential backoff + jitter (tenacity).
- A single run's exception is caught, recorded as `agent_error`, and the sweep continues.

### 5.6 LLM client (`llm.py`)
- Wrap **litellm** so one interface hits OpenAI, Anthropic, or local **Ollama** (`ollama/llama3.1`).
- Token + cost accounting from the provider response; cost=0 for Ollama.
- On-disk response cache (`diskcache`) keyed by `hash(model, prompt, temperature, messages)`
  for reproducibility and zero-cost re-runs.

### 5.7 Reporting (`report.py` + `dashboard.py`)
- `report.py` → Markdown table: per config → accuracy %, mean latency, mean cost/run,
  mean retries, and failure breakdown. Write to `results/REPORT.md` for the README.
- `dashboard.py` (Streamlit): leaderboard of configs, accuracy-by-difficulty, failure
  taxonomy bar chart, and a per-task drill-down (question, expected vs actual, final SQL).

---

## 6. Build plan (phases — execute in order)

### Phase 0 — Scaffold
- Deliverables: repo tree from §3, `pyproject.toml` (langgraph, litellm, duckdb, streamlit,
  pydantic, tenacity, diskcache, pytest, ruff), `Makefile` targets, `.gitignore`, empty modules.
- **Acceptance:** `make setup` installs; `ruff check` and `pytest` run green on empty tests.

### Phase 1 — Database
- Deliverables: `db.py` loads **Northwind** into DuckDB; `list_tables`/`get_schema` helpers;
  result canonicalization function with unit tests.
- **Acceptance:** `make db` produces `data/northwind.duckdb`; `test_db_canonical.py` passes
  (e.g. `[(2,1.0),(1,2.0)]` and `[(1,2.0),(2,1.0)]` compare equal unordered).

### Phase 2 — Agent under test
- Deliverables: `Agent` Protocol; LangGraph `text_to_sql` agent with the four tools and the
  self-correction loop; read-only guard on `run_sql`.
- **Acceptance:** agent answers 3 hand-questions correctly against Northwind; a deliberately
  bad question triggers ≥1 retry; a non-SELECT is rejected.

### Phase 3 — Golden task set
- Deliverables: `data/tasks.yaml` with **50 tasks** spanning easy/medium/hard and tags
  (aggregation, join, filter, ranking, subquery). Each has a verified `expected_result`
  (generate candidate SQL, execute it, snapshot the rows, then hand-verify a sample).
- **Acceptance:** a `make verify-tasks` script executes each `expected_sql` and confirms it
  returns the stored `expected_result`; 50/50 verified.

### Phase 4 — Runner + store
- Deliverables: `store.py` (SQLite DDL + upsert), `runner.py` (async, retry, timeout, cache),
  `llm.py`.
- **Acceptance:** `make eval CONFIG=configs/baseline.yaml` runs all 50 tasks, writes 50 rows
  to SQLite; a second run is idempotent (still 50 rows) and hits the cache (near-zero cost).

### Phase 5 — Scoring + classification
- Deliverables: `scorers.py` (execution accuracy), `classify.py` (taxonomy), unit tests.
- **Acceptance:** `test_scorers.py` and `test_classify.py` pass; every non-success run in the
  store has a non-null `failure_category` drawn from the fixed set.

### Phase 6 — Report + dashboard
- Deliverables: `report.py` → `results/REPORT.md`; `dashboard.py` Streamlit app.
- **Acceptance:** `make report` emits a Markdown table with accuracy/latency/cost/failures;
  `make dashboard` serves the leaderboard, failure chart, and per-task drill-down.

### Phase 7 — A/B sweep
- Deliverables: 4 configs (2 prompt versions × 2 models, e.g. gpt-4o-mini vs Claude Haiku).
  A `make sweep` target runs all configs and regenerates the comparison.
- **Acceptance:** `results/REPORT.md` shows all 4 configs side by side with a clear winner on
  accuracy and a cost/latency trade-off called out in prose.

### Phase 8 — CI + polish
- Deliverables: `.github/workflows/ci.yml` (ruff + pytest + a **5-task smoke eval** using
  Ollama or a mocked LLM so CI needs no paid key); README with the results table, an
  architecture diagram, and dashboard screenshots; MIT license.
- **Acceptance:** CI is green on a fresh clone; README renders the results table; repo is
  public and pinned.

---

## 7. Trade-offs (made explicit)

| Decision | Chosen | Alternative | Why / cost |
|---|---|---|---|
| Scoring | Execution accuracy (result match) | Exact-SQL match | Many SQLs → same rows; exact-match under-counts correct answers. Cost: must canonicalize result sets carefully. |
| Reference DB | Local DuckDB + Northwind | BigQuery sandbox; Chinook / TPC-H | Free, offline, reproducible, instant; a classic orders/products/customers business schema that maps to real analytics questions. Cost: less "cloud-scale" realism and a well-known toy schema (possible training-data exposure); note BigQuery as a documented swap. |
| Agent framework | LangGraph | Plain function-calling loop | Resume keyword + graph/checkpointing; agent is behind an interface so it stays swappable. Cost: extra dependency. |
| Store | SQLite | Parquet / DuckDB | Trivial querying for the dashboard, upserts for idempotency. Cost: not built for millions of runs (fine at this scale). |
| Classifier | Rule-based first, LLM fallback | Pure LLM judge | Deterministic, free, debuggable. Cost: rules need maintenance as new failure modes appear. |
| Providers | litellm (multi) + Ollama | Single SDK | Cheap A/B across models and a free local fallback for CI. Cost: one more abstraction. |
| Determinism | temp=0 + response cache | Sampling + multiple seeds | Reproducible, cheap re-runs, stable README numbers. Cost: hides run-to-run variance (add a seeded multi-run mode later). |

---

## 8. What to revisit as it grows
- **Variance:** add N-trials-per-task with mean±std once single-shot works (agents are stochastic).
- **Harder tasks:** graduate from Northwind to the Spider/BIRD text-to-SQL benchmarks for credibility.
- **Scale store:** move SQLite → DuckDB/Parquet if runs reach 10k+.
- **Tracing:** wire LangSmith or Arize Phoenix for span-level traces beyond local JSONL.
- **CI cost:** keep the smoke eval on a mocked/local model so CI never needs a paid key.
- **Generalize the AUT:** prove the harness is agent-agnostic by dropping in a second agent
  (e.g. a tool-using research agent) behind the same `Agent` Protocol.

---

## 9. Resume bullet this produces (fill the blanks after Phase 7)
> Built an open-source evaluation harness for LLM agents that scored a LangGraph
> text-to-SQL agent at **__% execution accuracy** across **50 golden tasks** and **4
> model/prompt configurations**, with a rule-based failure taxonomy and an A/B comparison
> surfacing a **__× cost difference** at **__pp** accuracy trade-off between models.

---

## Status
- 2026-11-15: Design finalized; build not yet started. Project `agent-eval-harness` moved to **building** on the weekend-project tracker.
- Open decisions before build: (1) A/B models — default gpt-4o-mini vs Claude Haiku, or all-local Ollama for zero cost; (2) keep Chinook for v1, graduate to Spider/BIRD later.
- 2026-11-15: Decisions — (1) A/B models: **gpt-4o-mini vs Claude Haiku**; (2) reference DB switched from Chinook to **Northwind** (Spider/BIRD still the later graduation path). Phase 0 scaffold complete.
- 2026-11-21: Phase 1 complete. Source = `pthom/northwind_psql` pinned to commit `4aa7967`, verified by SHA-256 and downloaded at build time rather than committed (the source repo has no clear license). Conversion: `real`→DOUBLE, empty `bytea` image columns dropped, PKs enforced, FKs stored as column comments for the agent's `get_schema`.
- 2026-11-21: Phase 2 complete. LangGraph model⇄tools loop; read-only guard via DuckDB's parser; LLM injected (scripted fake for tests/CI). Deviations: `max_iters` default 6→12 (a clean v1 run is ~6 turns); `AgentResult` gains `stop_reason`, `n_nudges`, `n_recovered_calls`; agent recovers JSON-as-text tool calls and nudges once if no query was ever run. Dev model: `ollama/llama3.1:8b` (`num_ctx` 8192). Live acceptance — v2 prompt: 3/3 hand questions, bad question → 2 retries, DELETE rejected; v1: 2/3 (revenue question fails on schema hallucination).
- 2026-11-21: Phase 3 complete. `data/tasks.yaml`: 50 answerable tasks (15 easy / 20 medium / 15 hard; tags incl. aggregation, join, filter, ranking, subquery, date, having, null_handling, window, self_join; 3 ordered). `golden.py` snapshot/verify/stats; `make verify-tasks` → 50/50 and runs in CI. Hand-verification: 12 tasks cross-checked against independently written SQL (12/12 agree); every ranking cutoff checked for ties. Lint rules: no empty results, ≤25 rows, known tags, ORDER BY required when `ordered`. Unanswerable "should decline" tasks deferred to a separate track (the agent's nudge would bias them).
