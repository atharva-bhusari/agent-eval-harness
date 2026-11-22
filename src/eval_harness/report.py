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
                # what one correct answer costs: penalizes cheap-but-wrong configs
                "cost_per_correct_usd": g["cost_usd"].sum() / k if k else float("nan"),
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


def mcnemar_exact(a_only: int, b_only: int) -> float:
    """Two-sided exact McNemar p-value: a binomial test on the discordant tasks only.

    Tasks both configs pass (or both fail) carry no information about which is better;
    under H0 each discordant task is a fair coin flip between "A won" and "B won".
    """
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def _pair_kind(a: pd.Series, b: pd.Series) -> str:
    if a["model"] == b["model"] and a["prompt_version"] != b["prompt_version"]:
        return "prompt"
    if a["prompt_version"] == b["prompt_version"] and a["model"] != b["model"]:
        return "model"
    return "model + prompt"


PAIR_COLUMNS = [
    "kind",
    "a",
    "b",
    "n",
    "both_pass",
    "a_only",
    "b_only",
    "both_fail",
    "acc_a",
    "acc_b",
    "delta",
    "p_value",
    "a_only_tasks",
    "b_only_tasks",
]


def paired_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    """Every pair of configs compared on the tasks *both* ran (paired, so task difficulty
    cancels out). Prompt pairs are oriented v1 -> v2; others by config id. Sorted with
    prompt effects first, then model effects, then the rest."""
    meta = df.groupby("config_id")[["model", "prompt_version"]].first()
    wide = df.pivot_table(index="task_id", columns="config_id", values="success", aggfunc="first")
    ids = sorted(meta.index)
    rows = []
    for i, x in enumerate(ids):
        for y in ids[i + 1 :]:
            a, b = x, y
            kind = _pair_kind(meta.loc[a], meta.loc[b])
            if kind == "prompt" and meta.at[a, "prompt_version"] > meta.at[b, "prompt_version"]:
                a, b = b, a
            both = wide[[a, b]].dropna()
            if both.empty:
                continue
            sa, sb = both[a].astype(bool), both[b].astype(bool)
            a_only, b_only = int((sa & ~sb).sum()), int((~sa & sb).sum())
            rows.append(
                {
                    "kind": kind,
                    "a": a,
                    "b": b,
                    "n": len(both),
                    "both_pass": int((sa & sb).sum()),
                    "a_only": a_only,
                    "b_only": b_only,
                    "both_fail": int((~sa & ~sb).sum()),
                    "acc_a": sa.mean(),
                    "acc_b": sb.mean(),
                    "delta": sb.mean() - sa.mean(),
                    "p_value": mcnemar_exact(a_only, b_only),
                    "a_only_tasks": ", ".join(sorted(both.index[sa & ~sb])),
                    "b_only_tasks": ", ".join(sorted(both.index[~sa & sb])),
                }
            )
    # Keep the columns even with no pairs (one config, or no shared tasks), so callers can
    # filter on them without special-casing an empty result.
    out = pd.DataFrame(rows, columns=PAIR_COLUMNS)
    if out.empty:
        return out
    order = {"prompt": 0, "model": 1, "model + prompt": 2}
    out["_o"] = out["kind"].map(order)
    return out.sort_values(["_o", "a", "b"]).drop(columns="_o").reset_index(drop=True)


ALPHA = 0.05


def _pair(pairs: pd.DataFrame, x: str, y: str) -> pd.Series | None:
    hit = pairs[((pairs["a"] == x) & (pairs["b"] == y)) | ((pairs["a"] == y) & (pairs["b"] == x))]
    return None if hit.empty else hit.iloc[0]


def _cheaper(cheap: float, dear: float) -> str:
    return "is free (local)" if cheap == 0 else f"is {dear / cheap:.1f}× cheaper per run"


def summary_prose(summary: pd.DataFrame, pairs: pd.DataFrame) -> list[str]:
    """Plain-language verdict: winner (and whether its lead is significant), the cost and
    latency trade-off, and each prompt effect. Generated from the numbers, deterministic."""
    full = summary[summary["n"] >= summary["n_tasks"]]
    pool = full if not full.empty else summary
    ranked = pool.sort_values(
        ["accuracy", "mean_cost_usd", "mean_latency_s", "config_id"],
        ascending=[False, True, True, True],
    )
    w = ranked.iloc[0]
    lines = [
        f"**Winner on accuracy: `{w.config_id}`** ({w.model}, prompt {w.prompt_version}) at "
        f"**{_pct(w.accuracy)}** ({w.successes}/{w.n})."
    ]
    if full.empty:
        lines[0] += " ⚠️ No config has run the full task set yet, so this is provisional."
    tied = ranked.iloc[1:][ranked.iloc[1:]["accuracy"] == w.accuracy]
    if not tied.empty:
        names = ", ".join(f"`{c}`" for c in tied["config_id"])
        lines.append(f"It ties {names} on accuracy and ranks first on lower cost, then latency.")
    behind = ranked[ranked["accuracy"] < w.accuracy]
    if not behind.empty:
        # Significance against the best config that actually scored lower (a tie has no
        # lead to test).
        r = behind.iloc[0]
        p = _pair(pairs, w.config_id, r.config_id)
        if p is not None and p.p_value < ALPHA:
            lines.append(
                f"It is significantly ahead of the best of the rest, `{r.config_id}` "
                f"({_pct(r.accuracy)}; paired McNemar p = {p.p_value:.3f})."
            )
        elif p is not None:
            lines.append(
                f"Its lead over the best of the rest, `{r.config_id}` ({_pct(r.accuracy)}), "
                f"is **not statistically significant** (paired McNemar p = {p.p_value:.2f}): on "
                f"{p.n} tasks, treat them as tied on accuracy and choose on cost and latency."
            )

    cheapest = pool.sort_values(["mean_cost_usd", "config_id"]).iloc[0]
    fastest = pool.sort_values(["mean_latency_s", "config_id"]).iloc[0]
    trade = []
    if cheapest.mean_cost_usd < w.mean_cost_usd:
        gap = (w.accuracy - cheapest.accuracy) * 100
        trade.append(
            f"`{cheapest.config_id}` {_cheaper(cheapest.mean_cost_usd, w.mean_cost_usd)} "
            f"({_cost(cheapest.mean_cost_usd)} vs {_cost(w.mean_cost_usd)}) for "
            f"{gap:.0f} points less accuracy"
        )
    if fastest.mean_latency_s < w.mean_latency_s:
        gap = (w.accuracy - fastest.accuracy) * 100
        trade.append(
            f"`{fastest.config_id}` is {w.mean_latency_s / fastest.mean_latency_s:.1f}× faster "
            f"({fastest.mean_latency_s:.1f} s vs {w.mean_latency_s:.1f} s per task) for "
            f"{gap:.0f} points less accuracy"
        )
    if trade:
        lines.append("**Trade-off:** " + "; ".join(trade) + ".")
    else:
        lines.append("It is also the cheapest and the fastest: no cost/latency trade-off.")
    if pd.notna(w.cost_per_correct_usd) and w.cost_per_correct_usd > 0:
        lines.append(f"Cost per correct answer for the winner: {_cost(w.cost_per_correct_usd)}.")

    for p in pairs[pairs["kind"] == "prompt"].itertuples():
        verdict = "significant" if p.p_value < ALPHA else "not significant"
        lines.append(
            f"Prompt effect on `{summary.set_index('config_id').at[p.a, 'model']}`: "
            f"`{p.a}` → `{p.b}` is {p.delta * 100:+.0f} points on {p.n} shared tasks "
            f"({p.b_only} fixed, {p.a_only} broken; p = {p.p_value:.3f}, {verdict})."
        )
    return lines


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


def render_markdown(df: pd.DataFrame, n_tasks: int, title: str = "Eval report") -> str:
    """Deterministic for a given store (no timestamps), so REPORT.md diffs are meaningful."""
    out = [f"# {title}", ""]
    if df.empty:
        return "\n".join([*out, "_No runs in the store yet. Run `make eval` first._", ""])

    summary = config_summary(df, n_tasks)
    pairs = paired_comparisons(df)
    out += [
        f"{len(summary)} config(s), {len(df)} runs, golden set of {n_tasks} tasks. "
        "Accuracy = execution accuracy (the agent's result set equals the expected rows). "
        "The interval is a 95% Wilson interval: at this sample size, configs whose intervals "
        "overlap heavily are not reliably different. Cost and latency are nominal (a cached "
        "re-run reports the original run's numbers).",
        "",
        "## Summary",
        "",
        *(f"- {line}" for line in summary_prose(summary, pairs)),
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
                "—" if pd.isna(r.cost_per_correct_usd) else _cost(r.cost_per_correct_usd),
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
                "cost/correct",
                "retries/run",
                "failures",
            ],
            rows,
            "lllrrrrrrrl",
        )
    )
    if (summary["n"] < summary["n_tasks"]).any():
        out += [
            "",
            "⚠️ = partial run: accuracy covers only the tasks that were run, so it is "
            "not comparable with full runs.",
        ]

    if not pairs.empty:
        out += [
            "",
            "## Head-to-head (paired)",
            "",
            "Each pair is compared only on the tasks both configs ran. *A only* / *B only* "
            "count tasks exactly one of them got right; the exact McNemar p-value tests "
            f"whether that split could be chance (significant below {ALPHA}).",
            "",
        ]
        rows = [
            [
                p.kind,
                f"`{p.a}`",
                f"`{p.b}`",
                p.n,
                f"{_pct(p.acc_a)} → {_pct(p.acc_b)}",
                f"{p.delta * 100:+.0f} pts",
                p.a_only,
                p.b_only,
                f"{p.p_value:.3f}" + (" ✔" if p.p_value < ALPHA else ""),
            ]
            for p in pairs.itertuples()
        ]
        out.append(
            _md_table(
                [
                    "differs in",
                    "A",
                    "B",
                    "shared tasks",
                    "accuracy A → B",
                    "Δ",
                    "A only",
                    "B only",
                    "p",
                ],
                rows,
                "lllrrrrrr",
            )
        )
        flips = [p for p in pairs.itertuples() if p.a_only_tasks or p.b_only_tasks]
        if flips:
            out += ["", "<details>", "<summary>Which tasks flipped</summary>", ""]
            for p in flips:
                out.append(
                    f"- `{p.a}` → `{p.b}`: fixed {p.b_only_tasks or '—'}; "
                    f"broken {p.a_only_tasks or '—'}"
                )
            out += ["", "</details>"]

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
    title: str = "Eval report",
) -> Path:
    df = load_runs(store_path, configs)
    if n_tasks is None:
        n_tasks = len(golden.load_tasks())
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown(df, n_tasks, title), encoding="utf-8")
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
