# Agent Eval Harness

A reusable **evaluation harness for LLM agents**. It runs an agent-under-test against a fixed
golden task set, captures a full trace of every run (tool calls, tokens, latency, cost), scores
each answer on **execution accuracy** (does it return the right rows, regardless of wording), classifies
every failure into a **taxonomy**, and compares **prompt versions × models** in a Markdown table
and a Streamlit dashboard.

The reference agent-under-test is a **LangGraph text-to-SQL agent** over a local DuckDB copy of
the Northwind database (a food import/export company: customers, orders, products, suppliers,
employees). It sits behind an `Agent` Protocol, so any agent can be swapped in.

> **Status:** 🚧 Phases 0–7 built (scaffold, Northwind DB + result canonicalization,
> LangGraph text-to-SQL agent, 50-task verified golden set, async runner + SQLite store +
> response cache, scorer + root-cause failure classifier, report + dashboard, A/B sweep with
> paired significance tests). The gpt-4o-mini half of the 2×2 grid has run; the Claude Haiku
> 5.5 half is waiting on an API key. Full report: [results/REPORT.md](results/REPORT.md).

## Results so far

Execution accuracy on the 50-task golden set (95% Wilson interval), from
[results/REPORT.md](results/REPORT.md) and the local runs:

| config | model | prompt | accuracy | mean latency | cost / run |
|---|---|---|---:|---:|---:|
| `gpt4o_mini_v2` | gpt-4o-mini | v2 | **92%** (81–97%) | 4.1 s | $0.0005 |
| `gpt4o_mini_v1` | gpt-4o-mini | v1 | 90% (79–96%) | 5.7 s | $0.0006 |
| `ollama_llama_v2` | llama3.1:8b (local) | v2 | 80% (67–89%) | 27.5 s | $0 |
| `haiku_v1` / `haiku_v2` | Claude Haiku 5.5 | v1 / v2 | pending | | |

- **The prompt's value depends on the model.** v2 puts the schema in the prompt. For
  gpt-4o-mini it adds only +2 points (2 tasks fixed, 1 broken; paired McNemar p = 1.0, so no real
  difference), but it is 28% faster and 17% cheaper per task, because the model skips schema discovery
  (2.3 tool calls per task instead of 5.2).
  For llama3.1:8b it is the difference between failing and working: without it, the model
  invents column names.
- **Hard tasks are the ceiling.** Both gpt-4o-mini configs score 100% on easy tasks and 73% on
  hard ones. The misses are real analyst mistakes, such as counting order lines instead of orders
  after a join, or swapping `DATEDIFF`'s arguments, and two different models make the same ones.
- Both gpt-4o-mini configs (100 runs) cost about **$0.06** in total. Re-running them costs $0,
  because responses are cached.

Try the agent on one question (free and local via [Ollama](https://ollama.com), after
`ollama pull llama3.1:8b`):

```bash
uv run python -m eval_harness.agent.text_to_sql "Which 5 customers placed the most orders?" --prompt v2
```

## Architecture

```
 configs/*.yaml ──┐
                  ▼
 tasks.yaml ──► RUNNER (async, retry, timeout, cache) ──► SCORERS (execution accuracy)
                  ▲   │                                         │
     tools        │   │ trace                                   ▼
 AGENT UNDER TEST ┘   ▼                                  FAILURE CLASSIFIER
 (LangGraph text-to-SQL)  METRICS STORE (SQLite) ◄──────────────┘
        │                     │
        ▼                     ├──► results/REPORT.md (Markdown table)
   DuckDB (Northwind)         └──► Streamlit dashboard
```

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) (it installs Python 3.11+ for you).

```bash
uv sync                                   # install
cp .env.example .env                      # add API keys (needed from the eval phase on)

make test        # or: uv run python tasks.py test      (Windows, no make)
make lint        # or: uv run python tasks.py lint
```

| Target | What it does | Phase |
|---|---|---|
| `setup` | install dependencies | 0 ✅ |
| `db` | download the pinned Northwind dump and build `data/northwind.duckdb` | 1 ✅ |
| `verify-tasks` | check every golden task's reference SQL reproduces its expected rows (50/50) | 3 ✅ |
| `eval CONFIG=…` | run all tasks for one config (`--tasks`, `--limit`, `--resume`) | 4 ✅ |
| `reclassify` | re-label stored failures after a classifier change (no model calls) | 5 ✅ |
| `report` | write `results/REPORT.md`: accuracy (95% CI), latency, cost, retries, failures | 6 ✅ |
| `dashboard` | Streamlit app: leaderboard, accuracy by difficulty, failure taxonomy, task × config grid, per-task drill-down | 6 ✅ |
| `sweep` | run a grid of configs (`configs/sweeps/`) and regenerate the report; skips configs without an API key, `--max-usd` caps spend | 7 ✅ |
| `smoke` | 5-task eval on a mocked LLM (CI) | 8 |

## Layout

```
configs/            model × prompt configs (A/B grid)
data/tasks.yaml     golden task set
src/eval_harness/   types · db · llm · agent/ · runner · scorers · classify · store · report · dashboard
tests/              unit + smoke tests
```

## License

MIT
