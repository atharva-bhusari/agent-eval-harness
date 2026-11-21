# Task targets. On Windows without make, use the mirror: `uv run python tasks.py <target>`.
CONFIG ?= configs/baseline.yaml

.PHONY: setup db verify-tasks eval sweep report dashboard lint format test smoke

setup:          ## install deps into .venv
	uv sync

db:             ## build data/northwind.duckdb
	@echo "not implemented (Phase 1)"

verify-tasks:   ## check every expected_sql reproduces expected_result
	@echo "not implemented (Phase 3)"

eval:           ## run all tasks for one config: make eval CONFIG=configs/baseline.yaml
	@echo "not implemented (Phase 4) - CONFIG=$(CONFIG)"

sweep:          ## run every config in configs/ and regenerate the report
	@echo "not implemented (Phase 7)"

report:         ## write results/REPORT.md
	@echo "not implemented (Phase 6)"

dashboard:      ## serve the Streamlit dashboard
	@echo "not implemented (Phase 6)"

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
