# Agent Eval Harness

A reusable **evaluation harness for LLM agents**. It runs an agent-under-test against a fixed
golden task set, captures a full trace of every run (tool calls, tokens, latency, cost), scores
each answer on **execution accuracy** (does it return the right rows, regardless of wording), classifies
every failure into a **taxonomy**, and compares **prompt versions × models** in a Markdown table
and a Streamlit dashboard.

The reference agent-under-test is a **LangGraph text-to-SQL agent** over a local DuckDB copy of
the Northwind database (a food import/export company: customers, orders, products, suppliers,
employees). It sits behind an `Agent` Protocol, so any agent can be swapped in.

> **Status:** 🚧 Phases 0–4 complete (scaffold, Northwind DB + result canonicalization,
> LangGraph text-to-SQL agent, 50-task verified golden set, async runner + SQLite store +
> response cache). Full results table lands after Phase 7.
>
> **First result:** llama3.1:8b (local, Ollama) with the v2 prompt scores **40/50 = 80%**
> execution accuracy, at 27.5 s mean latency and $0. A cached re-run takes 2 minutes instead of 23.

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
| `report` / `dashboard` | Markdown table / Streamlit app | 6 |
| `sweep` | run the 2×2 A/B grid and regenerate the report | 7 |
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
