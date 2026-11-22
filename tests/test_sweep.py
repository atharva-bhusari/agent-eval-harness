"""A/B sweep: grid loading, key/budget skipping, report restricted to the grid."""

import asyncio

import pytest

from eval_harness import golden, sweep
from eval_harness.report import write_report
from eval_harness.store import Store
from eval_harness.types import AgentResult, Config

TASKS = golden.load_tasks()[:6]


class FixedAgent:
    """Answers the first ``n_right`` tasks correctly (by golden order), the rest wrongly."""

    def __init__(self, n_right: int, cost: float = 0.001):
        self.right = {t.question for t in TASKS[:n_right]}
        self.expected = {t.question: t.expected_result for t in TASKS}
        self.cost = cost

    def run(self, question: str) -> AgentResult:
        rows = list(self.expected[question]) if question in self.right else [("wrong",)]
        return AgentResult(
            answer_rows=rows,
            final_sql="SELECT 1",
            tool_calls=[],
            n_retries=0,
            prompt_tokens=10,
            completion_tokens=2,
            cost_usd=self.cost,
            latency_ms=100,
            spent_usd=self.cost,
        )


def grid(*configs: Config) -> sweep.Grid:
    return sweep.Grid(name="t", title="Test sweep", configs=list(configs))


def cfg(cid, model="fake/m", prompt="v1"):
    return Config(id=cid, model=model, prompt_version=prompt)


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "runs.sqlite") as s:
        yield s


def test_real_grids_load_and_are_a_2x2():
    g = sweep.load_grid(sweep.DEFAULT_GRID)
    assert [c.id for c in g.configs] == ["gpt4o_mini_v1", "gpt4o_mini_v2", "haiku_v1", "haiku_v2"]
    assert {(c.model, c.prompt_version) for c in g.configs} == {
        (m, p) for m in ("openai/gpt-4o-mini", "anthropic/claude-haiku-5-5") for p in ("v1", "v2")
    }
    assert [c.id for c in sweep.load_grid(sweep.CONFIG_DIR / "sweeps" / "local.yaml").configs] == [
        "ollama_llama",
        "ollama_llama_v2",
    ]


def test_grid_validation(tmp_path):
    bad = tmp_path / "g.yaml"
    bad.write_text("configs: [gpt4o_mini_v1, gpt4o_mini_v1]\n")
    with pytest.raises(ValueError, match="duplicate"):
        sweep.load_grid(bad)
    bad.write_text("configs: []\n")
    with pytest.raises(ValueError, match="at least one"):
        sweep.load_grid(bad)
    bad.write_text("configs: [gpt4o_mini_v1]\nmodels: [x]\n")
    with pytest.raises(ValueError, match="unknown keys"):
        sweep.load_grid(bad)


def test_missing_provider_key_is_skipped(monkeypatch, store):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    g = grid(
        cfg("a", "openai/gpt-4o-mini"),
        cfg("b", "anthropic/claude-haiku-5-5"),
        cfg("c", "ollama/llama3.1:8b"),
    )
    res = asyncio.run(sweep.run_sweep(g, TASKS, store, agent_factory=lambda c: FixedAgent(3)))
    assert res.ran == ["a", "c"]
    assert res.skipped == {"b": "ANTHROPIC_API_KEY is not set"}
    assert store.count("b") == 0 and store.count("a") == len(TASKS)


def test_budget_stops_starting_new_configs(store):
    g = grid(cfg("a"), cfg("b"), cfg("c"))
    res = asyncio.run(
        sweep.run_sweep(
            g, TASKS, store, max_usd=0.01, agent_factory=lambda c: FixedAgent(3, cost=0.002)
        )
    )
    assert res.ran == ["a"]  # a spends 6 x 0.002 = 0.012 >= 0.01
    assert set(res.skipped) == {"b", "c"} and "budget" in res.skipped["b"]
    assert res.spent_usd == pytest.approx(0.012)


def test_sweep_report_covers_only_the_grid(store, tmp_path):
    agents = {"v1": FixedAgent(2), "v2": FixedAgent(6), "other": FixedAgent(1)}
    g = grid(cfg("m_v1", prompt="v1"), cfg("m_v2", prompt="v2"))
    asyncio.run(sweep.run_sweep(g, TASKS, store, agent_factory=lambda c: agents[c.prompt_version]))
    asyncio.run(
        sweep.run_sweep(grid(cfg("other")), TASKS, store, agent_factory=lambda c: agents["other"])
    )
    md = write_report(
        store.path, tmp_path / "R.md", ["m_v1", "m_v2"], n_tasks=len(TASKS), title="Test sweep"
    ).read_text(encoding="utf-8")
    assert md.startswith("# Test sweep")
    assert "`other`" not in md
    assert "**Winner on accuracy: `m_v2`**" in md
    assert "| prompt | `m_v1` | `m_v2` | 6 |" in md  # paired v1 -> v2 row
