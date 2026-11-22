"""Markdown report + the aggregation shared with the dashboard.

Reads the SQLite store (never re-runs anything) and writes ``results/REPORT.md``:
- one row per config: accuracy with a 95% Wilson interval, coverage of the current golden
  set, mean latency, mean cost/run, mean retries, failure breakdown;
- accuracy by difficulty;
- failure taxonomy counts per config;
- every failed run with its category and the classifier's reason.

Cost and latency are *nominal* (what the run cost when first executed; cache hits replay
it), so a cached re-run reports the same numbers as the original.

CLI:  python -m eval_harness.report [--store results/runs.sqlite] [--out results/REPORT.md]
                                    [--configs a,b]
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from eval_harness import golden
from eval_harness.settings import env, repo_root
from eval_harness.store import Store
from eval_harness.types import DIFFICULTIES, FAILURE_CATEGORIES

DEFAULT_REPORT_PATH = repo_root() / "results" / "REPORT.md"
Z_95 = 1.959964


def default_store_path() -> Path:
    """Resolved at call time (not import time) so tests/dashboard can point elsewhere."""
    return Path(env("EVAL_HARNESS_STORE", str(repo_root() / "results" / "runs.sqlite")))


# --------------------------------------------------------------------------------------
# Loading + aggregation (pure pandas; the dashboard reuses these)
# --------------------------------------------------------------------------------------
def load_runs(store_path: Path | str | None = None, configs: Sequence[str] | None = None):
    """All runs as a DataFrame (empty, with the right columns, if the store is empty)."""
    with Store(store_path or default_store_path()) as store:
        df = pd.DataFrame(store.runs())
    if df.empty:
        from eval_harness.store import RUN_COLUMNS

        return pd.DataFrame(columns=RUN_COLUMNS)
    df["success"] = df["success"].astype(bool)
    if configs:
        df = df[df["config_id"].isin(configs)]
    return df.reset_index(drop=True)


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """95% Wilson score interval for a proportion. Better than the normal approximation
    at small n and near 0%/100%, which is exactly where a 50-task eval lives."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def config_summary(df: pd.DataFrame, n_tasks: int | None = None) -> pd.DataFrame:
    """One row per config, sorted by accuracy (desc) then config id."""
    rows = []
    for cid, g in df.groupby("config_id", sort=True):
        n, k = len(g), int(g["success"].sum())
        lo, hi = wilson_interval(k, n)
        fails = g.loc[~g["success"], "failure_category"].value_counts()
        rows.append(
            {
                "config_id": cid,
                "model": g["model"].iloc[0],
                "prompt_version": g["prompt_version"].iloc[0],
                "n": n,
                "n_tasks": n_tasks or n,
                "successes": k,
                "accuracy": k / n,
                "ci_low": lo,
                "ci_high": hi,
                "mean_latency_s": g["latency_ms"].mean() / 1000,
                "mean_cost_usd": g["cost_usd"].mean(),
                "total_cost_usd": g["cost_usd"].sum(),
                "mean_retries": g["n_retries"].mean(),
                "mean_tool_calls": g["n_tool_calls"].mean(),
                "failures": ", ".join(f"{c} {v}" for c, v in fails.items()),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["accuracy", "config_id"], ascending=[False, True]).reset_index(
        drop=True
    )


def accuracy_by_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    """Long format: config_id, difficulty, n, successes, accuracy (difficulty in spec order)."""
    g = df.groupby(["config_id", "difficulty"])["success"].agg(n="size", successes="sum")
    g = g.reset_index()
    g["accuracy"] = g["successes"] / g["n"]
    order = {d: i for i, d in enumerate(DIFFICULTIES)}
    g["_o"] = g["difficulty"].map(order)
    return g.sort_values(["config_id", "_o"]).drop(columns="_o").reset_index(drop=True)


def failure_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Wide format: one row per failure category (taxonomy order, only those that occur),
    one column per config."""
    failed = df[~df["success"]]
    if failed.empty:
        return pd.DataFrame()
    t = pd.crosstab(failed["failure_category"], failed["config_id"])
    present = [c for c in FAILURE_CATEGORIES if c in t.index]
    return t.loc[present]


def failed_runs(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "config_id",
        "task_id",
        "difficulty",
        "failure_category",
        "failure_detail",
        "score_detail",
    ]
    return df.loc[~df["success"], cols].sort_values(["config_id", "task_id"]).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------------------
def _md_table(header: Sequence[str], rows: Sequence[Sequence[object]], align: str) -> str:
    """``align``: one char per column, 'l' or 'r'."""
    sep = ["---:" if a == "r" else "---" for a in align]
    esc = [[str(c).replace("|", "\\|").replace("\n", " ") for c in r] for r in rows]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in esc]
    return "\n".join(lines)


def _pct(x: float) -> str:
    return f"{x:.0%}" if x in (0, 1) else f"{x:.1%}"


def _cost(x: float) -> str:
    return "$0" if x == 0 else f"${x:.4f}"


def render_markdown(df: pd.DataFrame, n_tasks: int) -> str:
    """Deterministic for a given store (no timestamps), so REPORT.md diffs are meaningful."""
    out = ["# Eval report", ""]
    if df.empty:
        return "\n".join([*out, "_No runs in the store yet. Run `make eval` first._", ""])

    summary = config_summary(df, n_tasks)
    out += [
        f"{len(summary)} config(s), {len(df)} runs, golden set of {n_tasks} tasks. "
        "Accuracy = execution accuracy (the agent's result set equals the expected rows). "
        "The interval is a 95% Wilson interval: with ~50 tasks, configs whose intervals "
        "overlap heavily are not reliably different.",
        "",
    ]
    out += ["## Leaderboard", ""]
    rows = []
    for r in summary.itertuples():
        coverage = f"{r.n}/{r.n_tasks}" + (" ⚠️" if r.n < r.n_tasks else "")
        rows.append(
            [
                f"`{r.config_id}`",
                f"`{r.model}`",
                r.prompt_version,
                coverage,
                f"**{_pct(r.accuracy)}** ({r.successes}/{r.n})",
                f"{_pct(r.ci_low)}–{_pct(r.ci_high)}",
                f"{r.mean_latency_s:.1f} s",
                _cost(r.mean_cost_usd),
                f"{r.mean_retries:.2f}",
                r.failures or "—",
            ]
        )
    out.append(
        _md_table(
            [
                "config",
                "model",
                "prompt",
                "tasks",
                "accuracy",
                "95% CI",
                "mean latency",
                "cost/run",
                "retries/run",
                "failures",
            ],
            rows,
            "lllrrrrrrl",
        )
    )
    if (summary["n"] < summary["n_tasks"]).any():
        out += [
            "",
            "⚠️ = partial run: accuracy covers only the tasks that were run, so it is "
            "not comparable with full runs.",
        ]

    diff = accuracy_by_difficulty(df)
    configs = list(summary["config_id"])
    out += ["", "## Accuracy by difficulty", ""]
    rows = []
    for d in DIFFICULTIES:
        cells = [d]
        for c in configs:
            hit = diff[(diff["config_id"] == c) & (diff["difficulty"] == d)]
            cells.append(
                "—"
                if hit.empty
                else f"{_pct(hit['accuracy'].iloc[0])} ({int(hit['successes'].iloc[0])}/"
                f"{int(hit['n'].iloc[0])})"
            )
        rows.append(cells)
    out.append(
        _md_table(["difficulty", *(f"`{c}`" for c in configs)], rows, "l" + "r" * len(configs))
    )

    fc = failure_counts(df)
    out += ["", "## Failure taxonomy", ""]
    if fc.empty:
        out.append("No failures.")
    else:
        rows = [
            [f"`{cat}`", *(int(fc.at[cat, c]) if c in fc.columns else 0 for c in configs)]
            for cat in fc.index
        ]
        out.append(
            _md_table(["category", *(f"`{c}`" for c in configs)], rows, "l" + "r" * len(configs))
        )

    fr = failed_runs(df)
    if not fr.empty:
        out += [
            "",
            "## Failed runs",
            "",
            "<details>",
            f"<summary>{len(fr)} failed runs with the classifier's reason</summary>",
            "",
        ]
        rows = [
            [
                f"`{r.config_id}`",
                r.task_id,
                r.difficulty,
                f"`{r.failure_category}`",
                _clip(
                    r.score_detail
                    if r.failure_category in ("wrong_result", "empty_result")
                    else r.failure_detail or r.score_detail
                ),
            ]
            for r in fr.itertuples()
        ]
        out.append(_md_table(["config", "task", "difficulty", "category", "why"], rows, "lllll"))
        out += ["", "</details>"]
    out.append("")
    return "\n".join(out)


def _clip(text: object, n: int = 140) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def write_report(
    store_path: Path | str | None = None,
    out_path: Path | str = DEFAULT_REPORT_PATH,
    configs: Sequence[str] | None = None,
    n_tasks: int | None = None,
) -> Path:
    df = load_runs(store_path, configs)
    if n_tasks is None:
        n_tasks = len(golden.load_tasks())
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown(df, n_tasks), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m eval_harness.report")
    p.add_argument("--store", type=Path, default=None)
    p.add_argument("--out", type=Path, default=DEFAULT_REPORT_PATH)
    p.add_argument("--configs", help="comma-separated config ids (default: all in the store)")
    args = p.parse_args(argv)
    configs = [c.strip() for c in args.configs.split(",")] if args.configs else None
    out = write_report(args.store, args.out, configs)
    print(out.read_text(encoding="utf-8"))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
