"""A/B sweep: run every config in a grid, then regenerate the comparison report.

A grid file (``configs/sweeps/*.yaml``) names the configs to compare:

    name: hosted
    title: "A/B sweep: gpt-4o-mini vs Claude Haiku 5.5 x prompt v1/v2"
    configs: [gpt4o_mini_v1, gpt4o_mini_v2, haiku_v1, haiku_v2]

Configs run one after another (each with its own concurrency). A config whose provider
key is missing is skipped with a clear message rather than failing 50 runs. ``--max-usd``
stops starting new configs once actual spend this sweep reaches the cap (checked between
configs, so one config can overshoot it). Re-runs are served from the LLM response cache,
so re-running a finished sweep costs ~$0.

CLI:  python -m eval_harness.sweep [--grid configs/sweeps/hosted.yaml] [--resume]
                                   [--tasks e01,m02] [--limit N] [--max-usd 2.0]
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from eval_harness import golden
from eval_harness.config import load_config
from eval_harness.report import DEFAULT_REPORT_PATH, write_report
from eval_harness.runner import _progress_line, run_eval, select_tasks
from eval_harness.settings import env, repo_root
from eval_harness.store import DEFAULT_STORE_PATH, Store
from eval_harness.types import Config

CONFIG_DIR = repo_root() / "configs"
DEFAULT_GRID = CONFIG_DIR / "sweeps" / "hosted.yaml"

# litellm provider prefix -> env var holding its key (local models need none)
PROVIDER_KEYS = {"openai/": "OPENAI_API_KEY", "anthropic/": "ANTHROPIC_API_KEY"}


@dataclass
class Grid:
    name: str
    title: str
    configs: list[Config]


@dataclass
class SweepResult:
    ran: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)  # config id -> reason
    spent_usd: float = 0.0


def load_grid(path: Path | str) -> Grid:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    unknown = set(data) - {"name", "title", "configs"}
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    ids = data.get("configs") or []
    if not ids:
        raise ValueError(f"{path}: 'configs' must list at least one config id")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate config ids")
    configs = [load_config(CONFIG_DIR / f"{cid}.yaml") for cid in ids]
    name = data.get("name", path.stem)
    return Grid(name=name, title=data.get("title", f"A/B sweep: {name}"), configs=configs)


def missing_key(config: Config) -> str | None:
    """The env var this config needs but doesn't have, or None."""
    for prefix, var in PROVIDER_KEYS.items():
        if config.model.startswith(prefix) and not env(var, ""):
            return var
    return None


async def run_sweep(
    grid: Grid,
    tasks,
    store: Store,
    resume: bool = False,
    max_usd: float | None = None,
    agent_factory=None,
    on_record=None,
) -> SweepResult:
    """``agent_factory(config) -> Agent`` lets tests inject fakes; default builds the real
    text-to-SQL agent."""
    result = SweepResult()
    for config in grid.configs:
        if (var := missing_key(config)) is not None:
            result.skipped[config.id] = f"{var} is not set"
            continue
        if max_usd is not None and result.spent_usd >= max_usd:
            result.skipped[config.id] = f"budget reached (${result.spent_usd:.4f} >= ${max_usd})"
            continue
        agent = agent_factory(config) if agent_factory else None
        summary = await run_eval(
            config,
            tasks,
            store,
            agent=agent,
            resume=resume,
            on_record=(lambda i, n, r, c=config.id: on_record(c, i, n, r)) if on_record else None,
        )
        result.ran.append(config.id)
        result.spent_usd += sum(r.spent_usd for r in summary.records)
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m eval_harness.sweep")
    p.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    p.add_argument("--tasks", help="comma-separated task ids (default: all)")
    p.add_argument("--limit", type=int, help="only the first N tasks (cheap preview)")
    p.add_argument("--resume", action="store_true", help="skip runs already in the store")
    p.add_argument("--max-usd", type=float, help="stop starting configs past this spend")
    p.add_argument("--store", type=Path, default=DEFAULT_STORE_PATH)
    p.add_argument("--out", type=Path, default=DEFAULT_REPORT_PATH)
    args = p.parse_args(argv)

    grid = load_grid(args.grid)
    tasks = select_tasks(golden.load_tasks(), args.tasks, args.limit)
    print(f"sweep {grid.name}: {len(grid.configs)} configs x {len(tasks)} tasks", flush=True)
    for c in grid.configs:
        need = missing_key(c)
        print(
            f"  {c.id:<16} {c.model:<28} prompt {c.prompt_version}"
            + (f"   SKIP: {need} not set" if need else ""),
            flush=True,
        )

    def progress(cid: str, i: int, n: int, r) -> None:
        print(f"{cid:<16} {_progress_line(i, n, r)}", flush=True)

    with Store(args.store) as store:
        result = asyncio.run(
            run_sweep(
                grid, tasks, store, resume=args.resume, max_usd=args.max_usd, on_record=progress
            )
        )
    # Report on the whole grid (including configs finished in earlier sweeps), not on
    # every config that happens to be in the store.
    out = write_report(args.store, args.out, [c.id for c in grid.configs], title=grid.title)
    print(f"\nran: {', '.join(result.ran) or 'nothing'}")
    for cid, why in result.skipped.items():
        print(f"skipped {cid}: {why}")
    print(f"spent this sweep: ${result.spent_usd:.4f}")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
