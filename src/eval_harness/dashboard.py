"""Streamlit dashboard over the metrics store.

Sections: leaderboard -> accuracy by difficulty + failure taxonomy -> task x config grid
-> per-task drill-down (question, expected vs actual rows, final SQL, failure reason,
tool calls, full transcript). Read-only: it never runs the agent.

Run:  make dashboard   (or: uv run streamlit run src/eval_harness/dashboard.py)
Store path: $EVAL_HARNESS_STORE, else results/runs.sqlite (editable in the sidebar).
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from eval_harness import golden
from eval_harness.report import (
    accuracy_by_difficulty,
    config_summary,
    default_store_path,
    failure_counts,
    load_runs,
    paired_comparisons,
)
from eval_harness.types import DIFFICULTIES, FAILURE_CATEGORIES

# Categorical palette (dataviz reference instance), fixed slot order, light/dark steps.
# Colour follows the config id (sorted), never its rank, so filtering doesn't repaint.
PALETTE = {
    "light": [
        "#2a78d6",
        "#eb6834",
        "#1baf7a",
        "#eda100",
        "#e87ba4",
        "#008300",
        "#6250d6",
        "#e34948",
    ],
    "dark": [
        "#3987e5",
        "#d95926",
        "#199e70",
        "#c98500",
        "#d55181",
        "#0ca30c",
        "#9085e9",
        "#e66767",
    ],
}
MAX_SERIES = len(PALETTE["light"])
PASS, FAIL = "✓", "✗"


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load(path: str, mtime: float) -> pd.DataFrame:  # mtime: cache key only
    return load_runs(path)


@st.cache_data(show_spinner=False)
def _tasks() -> dict:
    return {t.id: t for t in golden.load_tasks()}


def _mtime(path: Path) -> float:
    wal = path.with_name(path.name + "-wal")
    return max((p.stat().st_mtime for p in (path, wal) if p.exists()), default=0.0)


def _theme() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:  # older Streamlit / bare mode
        return "light"


def _color_scale(all_configs: list[str]) -> alt.Scale:
    colors = PALETTE[_theme()]
    domain = sorted(all_configs)[:MAX_SERIES]
    return alt.Scale(domain=domain, range=colors[: len(domain)])


# --------------------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------------------
def leaderboard(df: pd.DataFrame, n_tasks: int) -> None:
    st.subheader("Leaderboard")
    s = config_summary(df, n_tasks)
    s["tasks"] = s["n"].astype(str) + "/" + s["n_tasks"].astype(str)
    s["95% CI"] = s.apply(lambda r: f"{r.ci_low:.0%} – {r.ci_high:.0%}", axis=1)
    s["accuracy"] = s["accuracy"] * 100  # ProgressColumn printf format wants 0-100
    st.dataframe(
        s[
            [
                "config_id",
                "model",
                "prompt_version",
                "tasks",
                "accuracy",
                "95% CI",
                "mean_latency_s",
                "mean_cost_usd",
                "mean_retries",
                "failures",
            ]
        ],
        hide_index=True,
        width="stretch",
        column_config={
            "config_id": "config",
            "prompt_version": "prompt",
            "accuracy": st.column_config.ProgressColumn(
                "accuracy", format="%.1f%%", min_value=0.0, max_value=100.0
            ),
            "mean_latency_s": st.column_config.NumberColumn("mean latency", format="%.1f s"),
            "mean_cost_usd": st.column_config.NumberColumn("cost / run", format="$%.4f"),
            "mean_retries": st.column_config.NumberColumn("retries / run", format="%.2f"),
        },
    )
    if (s["n"] < s["n_tasks"]).any():
        st.caption(
            "Partial runs (tasks < total) are not comparable with full runs. With ~50 tasks, "
            "configs whose 95% intervals overlap heavily are not reliably different."
        )


def difficulty_chart(df: pd.DataFrame, scale: alt.Scale) -> None:
    st.subheader("Accuracy by difficulty")
    d = accuracy_by_difficulty(df)
    chart = (
        alt.Chart(d)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X(
                "difficulty:N", sort=list(DIFFICULTIES), title=None, axis=alt.Axis(labelAngle=0)
            ),
            xOffset=alt.XOffset("config_id:N", scale=alt.Scale(paddingInner=0.08)),
            y=alt.Y(
                "accuracy:Q",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%", tickCount=5, grid=True),
                title=None,
            ),
            color=alt.Color(
                "config_id:N", scale=scale, title="config", legend=alt.Legend(orient="top")
            ),
            tooltip=[
                alt.Tooltip("config_id:N", title="config"),
                alt.Tooltip("difficulty:N"),
                alt.Tooltip("accuracy:Q", format=".0%"),
                alt.Tooltip("successes:Q", title="passed"),
                alt.Tooltip("n:Q", title="tasks"),
            ],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, width="stretch")
    with st.expander("Table"):
        t = d.assign(result=d["successes"].astype(str) + "/" + d["n"].astype(str))
        st.dataframe(
            t.pivot(index="difficulty", columns="config_id", values="result").reindex(
                [x for x in DIFFICULTIES if x in set(d["difficulty"])]
            ),
            width="stretch",
        )


def failure_chart(df: pd.DataFrame, scale: alt.Scale) -> None:
    st.subheader("Failure taxonomy")
    fc = failure_counts(df)
    if fc.empty:
        st.info("No failures.")
        return
    long = fc.reset_index().melt(
        id_vars="failure_category", var_name="config_id", value_name="runs"
    )
    long = long[long["runs"] > 0]
    order = [c for c in FAILURE_CATEGORIES if c in set(long["failure_category"])]
    chart = (
        alt.Chart(long)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            y=alt.Y("failure_category:N", sort=order, title=None, axis=alt.Axis(labelLimit=220)),
            yOffset=alt.YOffset("config_id:N", scale=alt.Scale(paddingInner=0.08)),
            x=alt.X("runs:Q", title="failed runs", axis=alt.Axis(tickMinStep=1, grid=True)),
            color=alt.Color(
                "config_id:N", scale=scale, title="config", legend=alt.Legend(orient="top")
            ),
            tooltip=[
                alt.Tooltip("config_id:N", title="config"),
                alt.Tooltip("failure_category:N", title="category"),
                alt.Tooltip("runs:Q", title="failed runs"),
            ],
        )
        .properties(height=alt.Step(16))  # per-bar step: yOffset ignores a pixel height
    )
    st.altair_chart(chart, width="stretch")
    with st.expander("Table"):
        st.dataframe(fc, width="stretch")


def tradeoff_chart(df: pd.DataFrame, n_tasks: int, scale: alt.Scale) -> None:
    st.subheader("Accuracy vs cost / latency")
    axis = (
        st.segmented_control(
            "x axis",
            ["cost / run", "latency / task"],
            default="cost / run",
            label_visibility="collapsed",
        )
        or "cost / run"
    )
    s = config_summary(df, n_tasks)
    field, fmt, title = (
        ("mean_cost_usd", "$.4f", "mean cost per run (USD)")
        if axis == "cost / run"
        else ("mean_latency_s", ".1f", "mean latency per task (s)")
    )
    base = alt.Chart(s).encode(
        x=alt.X(
            f"{field}:Q",
            title=title,
            axis=alt.Axis(format=fmt, grid=True),
            scale=alt.Scale(zero=True, nice=True),
        ),
        y=alt.Y(
            "accuracy:Q",
            title=None,
            axis=alt.Axis(format="%", grid=True),
            scale=alt.Scale(domain=[0, 1]),
        ),
    )
    points = base.mark_point(filled=True, size=140, opacity=1).encode(
        color=alt.Color(
            "config_id:N", scale=scale, title="config", legend=alt.Legend(orient="top")
        ),
        tooltip=[
            alt.Tooltip("config_id:N", title="config"),
            alt.Tooltip("accuracy:Q", format=".1%"),
            alt.Tooltip("mean_cost_usd:Q", title="cost / run", format="$.4f"),
            alt.Tooltip("mean_latency_s:Q", title="latency (s)", format=".1f"),
            alt.Tooltip("n:Q", title="tasks run"),
        ],
    )
    # Direct labels: with >3 series, colour alone isn't enough to tell points apart. The
    # label stays in the theme's text colour; the dot beside it carries the identity.
    labels = base.mark_text(align="left", dx=9, dy=-9, fontSize=12).encode(text="config_id:N")
    st.altair_chart((points + labels).properties(height=300), width="stretch")
    st.caption("Up and to the left is better. Partial runs (fewer tasks) are not comparable.")


def head_to_head(df: pd.DataFrame) -> None:
    st.subheader("Head-to-head (paired)")
    pairs = paired_comparisons(df)
    if pairs.empty:
        st.info("Needs at least two configs that share tasks.")
        return
    view = pairs.assign(
        acc=pairs.apply(lambda p: f"{p.acc_a:.0%} → {p.acc_b:.0%}", axis=1),
        delta_pts=pairs["delta"] * 100,
        verdict=pairs["p_value"].map(lambda p: "significant" if p < 0.05 else "not significant"),
    )
    st.dataframe(
        view[
            [
                "kind",
                "a",
                "b",
                "n",
                "acc",
                "delta_pts",
                "a_only",
                "b_only",
                "p_value",
                "verdict",
                "b_only_tasks",
                "a_only_tasks",
            ]
        ],
        hide_index=True,
        width="stretch",
        column_config={
            "kind": "differs in",
            "a": "A",
            "b": "B",
            "n": st.column_config.NumberColumn("shared tasks"),
            "acc": "accuracy A → B",
            "delta_pts": st.column_config.NumberColumn("Δ (pts)", format="%+.0f"),
            "a_only": st.column_config.NumberColumn("A only"),
            "b_only": st.column_config.NumberColumn("B only"),
            "p_value": st.column_config.NumberColumn("p (McNemar)", format="%.3f"),
            "b_only_tasks": "fixed by B",
            "a_only_tasks": "broken by B",
        },
    )
    st.caption(
        "Compared only on tasks both configs ran. A only / B only = tasks exactly one got "
        "right; the exact McNemar test asks whether that split could be chance."
    )


def task_grid(df: pd.DataFrame, tasks: dict) -> None:
    st.subheader("Tasks × configs")
    only_fail = st.toggle("Only tasks that failed somewhere", value=True)
    cell = df.apply(lambda r: PASS if r.success else f"{FAIL} {r.failure_category}", axis=1)
    grid = df.assign(cell=cell).pivot(index="task_id", columns="config_id", values="cell")
    grid = grid.reindex(
        [t for t in tasks if t in grid.index] + [t for t in grid.index if t not in tasks]
    )
    if only_fail:
        grid = grid[grid.apply(lambda r: r.astype(str).str.startswith(FAIL).any(), axis=1)]
    grid.insert(0, "difficulty", [tasks[t].difficulty if t in tasks else "?" for t in grid.index])
    grid.insert(1, "question", [tasks[t].question if t in tasks else "" for t in grid.index])
    st.dataframe(grid.fillna("—"), width="stretch", height=min(35 * (len(grid) + 1) + 2, 520))
    st.caption(f"{PASS} passed · {FAIL} failed (category) · — not run")


def _val(x):
    """SQLite NULL can surface as None or NaN in pandas; NaN is truthy, so normalize."""
    return None if x is None or (isinstance(x, float) and pd.isna(x)) else x


def _rows_frame(rows: list | None) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([list(r) for r in rows])


def drill_down(df: pd.DataFrame, tasks: dict, store_path: Path) -> None:
    st.subheader("Drill-down")
    c1, c2 = st.columns(2)
    configs = sorted(df["config_id"].unique())
    cfg = c1.selectbox("Config", configs)
    sub = df[df["config_id"] == cfg].set_index("task_id")
    labels = {
        tid: f"{tid}  {PASS if r.success else FAIL + ' ' + r.failure_category}"
        for tid, r in sub.iterrows()
    }
    order = [t for t in tasks if t in labels] + [t for t in labels if t not in tasks]
    failed_first = sorted(order, key=lambda t: bool(sub.at[t, "success"]))
    tid = c2.selectbox("Task", failed_first, format_func=labels.get)
    if tid is None:
        return
    run = sub.loc[tid]
    task = tasks.get(tid)

    if task is not None:
        st.markdown(
            f"**{tid}** · {task.difficulty} · {', '.join(task.tags)}"
            + (" · ordered" if task.ordered else "")
        )
        st.markdown(f"> {task.question}")
    if run["success"]:
        st.success(f"{PASS} Passed: {run['score_detail']}")
    else:
        st.error(f"{FAIL} **{run['failure_category']}**: {run['score_detail']}")
        if _val(run.get("failure_detail")):
            st.caption(f"Classifier: {run['failure_detail']}")

    m = st.columns(5)
    m[0].metric("latency", f"{run['latency_ms'] / 1000:.1f} s")
    m[1].metric("tool calls", int(run["n_tool_calls"]))
    m[2].metric("retries", int(run["n_retries"]))
    m[3].metric(
        "tokens in / out", f"{run['prompt_tokens'] / 1000:.1f}k / {int(run['completion_tokens'])}"
    )
    m[4].metric("cost", f"${run['cost_usd']:.4f}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Expected rows**")
        st.dataframe(
            _rows_frame(task.expected_result if task else None), width="stretch", hide_index=True
        )
    with right:
        st.markdown("**Agent's rows**")
        answer = json.loads(_val(run["answer_json"])) if _val(run["answer_json"]) else None
        if answer is None:
            st.info("No result set.")
        else:
            st.dataframe(_rows_frame(answer), width="stretch", hide_index=True)

    st.markdown("**Agent's final SQL**")
    st.code(_val(run["final_sql"]) or "-- none", language="sql")
    if task is not None and task.expected_sql:
        with st.expander("Reference SQL"):
            st.code(task.expected_sql, language="sql")

    from eval_harness.store import Store

    with Store(store_path) as store:
        calls = store.tool_calls(run["run_id"])
    with st.expander(f"Tool calls ({len(calls)})"):
        if calls:
            st.dataframe(
                pd.DataFrame(calls)[["seq", "name", "ok", "error", "args_json"]],
                hide_index=True,
                width="stretch",
                column_config={"ok": st.column_config.CheckboxColumn("ok")},
            )
    with st.expander("Transcript"):
        _transcript(_val(run["transcript_json"]))


def _transcript(raw: str | None) -> None:
    if not raw:
        st.write("No transcript stored.")
        return
    for msg in json.loads(raw):
        role = msg.get("role", "?")
        content = msg.get("content") or ""
        name = f" · {msg['name']}" if msg.get("name") else ""
        st.markdown(f"**{role}{name}**")
        if role == "system":
            st.caption(f"system prompt ({len(content)} chars)")
        elif content:
            st.text(content if len(content) < 4000 else content[:4000] + " …")
        for call in msg.get("tool_calls") or []:
            fn = call.get("function", {})
            st.code(f"{fn.get('name')}({fn.get('arguments')})", language="text")


# --------------------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Agent Eval Harness", page_icon="📊", layout="wide")
    st.title("Agent Eval Harness")

    with st.sidebar:
        store_path = Path(st.text_input("Metrics store", str(default_store_path())))
        if not store_path.exists():
            st.error("Store not found. Run `make eval` first.")
            st.stop()
        df_all = _load(str(store_path), _mtime(store_path))
        all_configs = sorted(df_all["config_id"].unique())
        chosen = st.multiselect("Configs", all_configs, default=all_configs)
        if st.button("Reload"):
            st.cache_data.clear()
            st.rerun()

    if df_all.empty:
        st.info("The store has no runs yet. Run `make eval CONFIG=...` first.")
        return
    if not chosen:
        st.info("Pick at least one config in the sidebar.")
        return
    if len(chosen) > MAX_SERIES:
        st.warning(
            f"Showing the first {MAX_SERIES} configs: more than that can't be told apart by colour."
        )
        chosen = chosen[:MAX_SERIES]

    tasks = _tasks()
    df = df_all[df_all["config_id"].isin(chosen)]
    st.caption(
        f"{len(chosen)} config(s) · {len(df)} runs · {len(tasks)} golden tasks · "
        "accuracy = execution accuracy · cost and latency are nominal (cache hits replay "
        "the original run's numbers)"
    )
    scale = _color_scale(all_configs if len(all_configs) <= MAX_SERIES else chosen)

    leaderboard(df, len(tasks))
    left, right = st.columns(2)
    with left:
        difficulty_chart(df, scale)
    with right:
        failure_chart(df, scale)
    left, right = st.columns(2)
    with left:
        tradeoff_chart(df, len(tasks), scale)
    with right:
        head_to_head(df)
    task_grid(df, tasks)
    drill_down(df, tasks, store_path)


main()
