"""Plotly charts for the dashboard, built from the tidy frames the reports already write.

Inputs are ``data/metrics_long.csv`` (one row per arm x phase x metric, with a
``direction`` column) and ``data/paired_stats.csv`` (one row per contrast x metric).
Nothing is recomputed from raw runs. Arms sit on an axis rather than in a legend, so
colour has three jobs only: the baseline (muted), the other arms, and the best arm.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..interactive.plotly_util import apply_theme, has_plotly, role, to_div


def _div(fig) -> str:
    return to_div(fig)


def _roles(marker: list[str], ring: list[str] | str) -> dict:
    """Marker colours by token role, kept on the trace so the page's theme switch can restyle them."""
    rings = [ring] * len(marker) if isinstance(ring, str) else ring
    return {
        "marker": {
            "color": [role(r) for r in marker],
            "line": {"color": [role(r) for r in rings], "width": 2},
        },
        "meta": {"marker_roles": marker, "ring_roles": rings},
    }


HIGHER, LOWER = "higher", "lower"


def directions(long_df: pd.DataFrame) -> dict[str, str]:
    """metric -> ``higher``/``lower`` from the ``direction`` column; undeclared metrics are left out."""
    if "direction" not in long_df:
        return {}
    d = (
        long_df.dropna(subset=["direction"])
        .drop_duplicates("metric")
        .set_index("metric")["direction"]
    )
    return {m: v for m, v in d.items() if v in (HIGHER, LOWER)}


def rank_table(long_df: pd.DataFrame, phase: str, metrics: list[str] | None = None) -> pd.DataFrame:
    """arm x metric mean-ranks (1 = best, ties share), ordered best mean rank first."""
    dirs = directions(long_df)
    keep = [m for m in (metrics or dirs) if m in dirs]
    sub = long_df[(long_df["phase"] == phase) & long_df["metric"].isin(keep)]
    wide = sub.pivot_table(index="arm", columns="metric", values="mean", aggfunc="first")
    ranks = pd.DataFrame(
        {m: wide[m].rank(ascending=dirs[m] == LOWER, method="average") for m in wide.columns},
        index=wide.index,
    )
    return ranks.loc[ranks.mean(axis=1).sort_values(kind="stable").index]


def best_arm(long_df: pd.DataFrame, phase: str, metric: str) -> str | None:
    dirs = directions(long_df)
    sub = long_df[(long_df["phase"] == phase) & (long_df["metric"] == metric)].dropna(
        subset=["mean"]
    )
    if sub.empty or metric not in dirs:
        return None
    idx = sub["mean"].idxmax() if dirs[metric] == HIGHER else sub["mean"].idxmin()
    return str(sub.loc[idx, "arm"])


def _arm_roles(arms: list[str], baseline: str | None, best: str | None) -> list[str]:
    return ["baseline" if a == baseline else "accent" if a == best else "series" for a in arms]


def leaderboard_div(
    long_df: pd.DataFrame, *, phase: str, metrics: list[str], baseline: str | None
) -> str | None:
    """Dot-and-interval small multiples: one column per metric, arms on a shared y-axis."""
    if not has_plotly() or long_df.empty:
        return None
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    sub = long_df[long_df["phase"] == phase]
    metrics = [m for m in metrics if m in set(sub["metric"])][:4]
    if not metrics:
        return None
    dirs = directions(long_df)
    order = rank_table(long_df, phase, metrics).index.tolist() or sorted(sub["arm"].unique())
    order = [a for a in order if a in set(sub["arm"])] + sorted(set(sub["arm"]) - set(order))
    titles = [
        f"{m} ({'higher' if dirs.get(m) == HIGHER else 'lower' if dirs.get(m) == LOWER else 'no'} is better)"
        for m in metrics
    ]
    fig = make_subplots(
        rows=1, cols=len(metrics), shared_yaxes=True, subplot_titles=titles, horizontal_spacing=0.04
    )
    for j, m in enumerate(metrics, start=1):
        rows = sub[sub["metric"] == m].set_index("arm").reindex(order)
        n = rows["n"].astype(float)
        spread = rows["std"].where(n > 1)  # one SD across the rows each arm logged (unit below)
        unit = (
            rows["unit"].fillna("rows") if "unit" in rows else pd.Series("rows", index=rows.index)
        )
        best = best_arm(long_df, phase, m)
        styled = _roles(_arm_roles(order, baseline, best), "surface")
        styled["marker"]["size"] = 10
        fig.add_trace(
            go.Scatter(
                x=rows["mean"],
                y=order,
                mode="markers",
                meta=styled["meta"],
                marker=styled["marker"],
                error_x={
                    "type": "data",
                    "array": spread,
                    "visible": True,
                    "thickness": 1.5,
                    "width": 0,
                    "color": role("rule_strong"),
                },
                customdata=np.stack([rows["std"], n, unit], axis=-1),
                hovertemplate="%{y}<br>mean %{x:.4g}<br>SD %{customdata[0]:.3g} over %{customdata[1]:.0f} "
                "%{customdata[2]}"
                "<extra>" + m + "</extra>",
                showlegend=False,
            ),
            row=1,
            col=j,
        )
    fig.update_yaxes(autorange="reversed", automargin=True)
    fig.update_annotations(font={"size": 12, "color": role("ink_2")})
    apply_theme(fig, height=34 * len(order) + 110)
    return _div(fig)


def rank_heatmap_div(long_df: pd.DataFrame, *, phase: str, max_metrics: int = 30) -> str | None:
    """Arms x metrics, cell = rank among arms (1 = best, darkest), single-hue ramp."""
    if not has_plotly():
        return None
    import plotly.graph_objects as go

    from spectramr.core.design_tokens import SEQUENTIAL_BLUE

    ranks = rank_table(long_df, phase)
    if ranks.empty or ranks.shape[0] < 2:
        return None
    ranks = ranks.iloc[:, :max_metrics]
    n = ranks.shape[0]
    scale = [[i / (len(SEQUENTIAL_BLUE) - 1), c] for i, c in enumerate(reversed(SEQUENTIAL_BLUE))]
    fig = go.Figure(
        go.Heatmap(
            z=ranks.to_numpy(),
            x=list(ranks.columns),
            y=list(ranks.index),
            zmin=1,
            zmax=n,
            colorscale=scale,
            xgap=2,
            ygap=2,
            colorbar={"title": {"text": "rank"}, "tickvals": [1, n], "ticktext": ["best", "worst"]},
            hovertemplate="%{y}<br>%{x}: rank %{z:.1f} of " + str(n) + "<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed", automargin=True, showgrid=False)
    fig.update_xaxes(tickangle=-40, automargin=True, showgrid=False)
    apply_theme(fig, height=26 * n + 200)
    return _div(fig)


def oriented_effects(
    stats: pd.DataFrame, dirs: dict[str, str], *, alpha: float = 0.05
) -> pd.DataFrame:
    """Cohen's d per contrast row, signed so positive = the variant beats the baseline.

    The CI is the mean-difference CI rescaled by d / mean_diff (same paired SD), so all
    metrics share one unitless axis. Metrics with no declared direction are dropped.
    """
    need = {"metric", "variant", "mean_diff", "ci_low", "ci_high", "cohens_d"}
    if stats.empty or not need <= set(stats.columns):
        return pd.DataFrame()
    s = stats.copy()
    # A test run on a transformed scale (e.g. "1/N_eq (lower is better)") declares its own
    # direction, which overrides the metric's: the mean difference is on that scale.
    scale = s["scale"].astype(str) if "scale" in s else pd.Series("", index=s.index)
    declared = np.select(
        [scale.str.contains("lower is better"), scale.str.contains("higher is better")],
        [LOWER, HIGHER],
        default="",
    )
    s["direction"] = np.where(declared != "", declared, s["metric"].map(dirs).fillna(""))
    s = s[s["direction"] != ""]
    sign = np.where(s["direction"] == HIGHER, 1.0, -1.0)
    k = np.where(s["mean_diff"] != 0, s["cohens_d"] / s["mean_diff"], np.nan)
    lo, hi = s["ci_low"] * k * sign, s["ci_high"] * k * sign
    s["d"] = s["cohens_d"] * sign
    s["lo"], s["hi"] = np.minimum(lo, hi), np.maximum(lo, hi)
    p = s["p_holm"] if "p_holm" in s else pd.Series(np.nan, index=s.index)
    s["significant"] = p < alpha
    return s


def effect_dots_div(effects: pd.DataFrame, *, rows: str, metrics: list[str]) -> str | None:
    """Oriented effect sizes, one column per metric; filled = Holm-significant."""
    if not has_plotly() or effects.empty:
        return None
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    metrics = [m for m in metrics if m in set(effects["metric"])][:4]
    if not metrics:
        return None
    labels = list(dict.fromkeys(effects[rows]))
    fig = make_subplots(
        rows=1,
        cols=len(metrics),
        shared_yaxes=True,
        subplot_titles=metrics,
        horizontal_spacing=0.04,
    )
    for j, m in enumerate(metrics, start=1):
        e = effects[effects["metric"] == m].set_index(rows).reindex(labels)
        sig = e["significant"].fillna(False).astype(bool)
        styled = _roles(
            ["series" if v else "surface" for v in sig],
            ["series" if v else "baseline" for v in sig],
        )
        styled["marker"]["size"] = 10
        fig.add_trace(
            go.Scatter(
                x=e["d"],
                y=labels,
                mode="markers",
                meta=styled["meta"],
                marker=styled["marker"],
                error_x={
                    "type": "data",
                    "symmetric": False,
                    "array": e["hi"] - e["d"],
                    "arrayminus": e["d"] - e["lo"],
                    "thickness": 1.5,
                    "width": 0,
                    "color": role("rule_strong"),
                },
                customdata=np.stack(
                    [e.get("p_holm", pd.Series(np.nan, index=e.index)), e["mean_diff"]], axis=-1
                ),
                hovertemplate="%{y}<br>d = %{x:.2f}<br>Δ = %{customdata[1]:.3g}, Holm p = %{customdata[0]:.2g}"
                "<extra>" + m + "</extra>",
                showlegend=False,
            ),
            row=1,
            col=j,
        )
        fig.add_vline(x=0, line={"color": role("rule_strong"), "width": 1}, row=1, col=j)
    fig.update_yaxes(autorange="reversed", automargin=True)
    fig.update_xaxes(title={"text": "effect size d (right = better)", "font": {"size": 11}})
    fig.update_annotations(font={"size": 12, "color": role("ink_2")})
    apply_theme(fig, height=34 * len(labels) + 130)
    return _div(fig)
