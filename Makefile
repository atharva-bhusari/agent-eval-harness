# Task targets. On Windows without make, use the mirror: `uv run python tasks.py <target>`.
CONFIG ?= configs/baseline.yaml

.PHONY: setup db verify-tasks eval reclassify sweep report dashboard lint format test smoke

setup:          ## install deps into .venv
	uv sync

db:             ## build data/northwind.duckdb (downloads the pinned Northwind dump)
	uv run python -m eval_harness.db build

verify-tasks:   ## check every expected_sql reproduces expected_result (+ lint rules)
	uv run python -m eval_harness.golden verify

eval:           ## run all tasks for one config: make eval CONFIG=configs/baseline.yaml [ARGS="--resume"]
	uv run python -m eval_harness.runner --config $(CONFIG) $(ARGS)

reclassify:     ## re-label stored failed runs after a classifier change (no model calls) [ARGS="--dry-run"]
	uv run python -m eval_harness.classify reclassify $(ARGS)

sweep:          ## run every config in configs/ and regenerate the report
	@echo "not implemented (Phase 7)"

report:         ## write results/REPORT.md from the store [ARGS="--configs a,b"]
	uv run python -m eval_harness.report $(ARGS)

dashboard:      ## serve the Streamlit dashboard (http://localhost:8501)
	uv run streamlit run src/eval_harness/dashboard.py

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

test:
	uv run pytest

smoke:          ## 5-task eval with a mocked LLM (used in CI)
	@echo "not implemented (Phase 8)"
