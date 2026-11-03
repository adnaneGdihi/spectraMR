r"""Interactive (plotly) twins of the high-value static report figures.

Each ``*_div`` returns a self-contained inline ``<div>`` string (plotly.js is
injected once by the HTML assembler, not here) or ``None`` when plotly is absent
or the data needed is missing — mirroring the static plotters' soft-skip so the
report can fall back to the embedded PNG cleanly.

Builders
--------
- :func:`group_iqm_div` — compact box+strip of every metric at once (overview).
- :func:`learning_curve_div` — training/validation dynamics with a metric
  dropdown + log-y toggle.
- :func:`metric_distribution_div` — per-metric distribution (raincloud) with a
  metric dropdown, split by method when several are present.
- :func:`comparison_div` — Bland-Altman agreement between two methods.
"""

from __future__ import annotations

import pandas as pd

from ..legend import arm_palette
from .plotly_util import apply_theme, caption, flag_colour, has_plotly, role, seat_menus, to_div

_NON_METRIC = {
    "case_id",
    "split",
    "step",
    "subject_id",
    "method",
    "run_id",
    "metric",
    "value",
    "epoch",
    "iteration",
    "batch_index",
}

# Arms keep the colour the static figures give them (one owner: legend.arm_palette).
_SURFACE_LINE = role("surface")
_MUTED_LINE = role("muted")


def _directions() -> dict[str, bool]:
    """Map metric -> higher-is-better, or empty when the registry is unavailable."""
    try:
        from spectramr.core.metrics.metric_directions import METRIC_HIGHER_IS_BETTER
    except Exception:
        return {}
    return dict(METRIC_HIGHER_IS_BETTER)


def _arrow(metric: str, directions: dict[str, bool]) -> str:
    base = (
        metric.split("_", 1)[-1] if metric.split("_", 1)[0] in {"train", "val", "test"} else metric
    )
    if base not in directions:
        return metric
    return f"{metric} ({'↑' if directions[base] else '↓'} better)"


def _series_from_wide(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """metric -> frame(value, label) from a wide per-case frame."""
    labels = (
        df["case_id"].astype(str)
        if "case_id" in df.columns
        else pd.Series([str(i) for i in range(len(df))])
    )
    out: dict[str, pd.DataFrame] = {}
    for col in df.columns:
        if col in _NON_METRIC or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        frame = pd.DataFrame({"value": vals, "label": labels}).dropna(subset=["value"])
        if not frame.empty:
            out[col] = frame
    return out


def _series_from_long(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """metric -> frame(value, label) from a long predictions frame."""
    if not {"metric", "value"}.issubset(df.columns):
        return {}
    label_col = "subject_id" if "subject_id" in df.columns else None
    out: dict[str, pd.DataFrame] = {}
    for metric, grp in df.groupby("metric"):
        vals = pd.to_numeric(grp["value"], errors="coerce")
        labels = (
            grp[label_col].astype(str)
            if label_col
            else pd.Series([str(i) for i in range(len(grp))], index=grp.index)
        )
        frame = pd.DataFrame({"value": vals, "label": labels}).dropna(subset=["value"])
        if not frame.empty:
            out[str(metric)] = frame
    return out


def _collect_series(per_call_df, predictions_df) -> dict[str, pd.DataFrame]:
    if per_call_df is not None and not per_call_df.empty:
        series = _series_from_wide(per_call_df)
        if series:
            return series
    if predictions_df is not None and not predictions_df.empty:
        return _series_from_long(predictions_df)
    return {}


# Above this many cases per metric the page would carry every row as JSON (13.5k cases x
# 36 metrics was 17 MB per chart), so dots are a seeded sample that keeps every outlier.
MAX_POINTS = 400


def _box_stats(values: pd.Series) -> dict[str, float]:
    """Exact Tukey box statistics over every case (whiskers at the last point within 1.5 IQR)."""
    v = values.to_numpy(dtype=float)
    q1, med, q3 = (float(x) for x in pd.Series(v).quantile([0.25, 0.5, 0.75]))
    lo_f, hi_f = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    inside = v[(v >= lo_f) & (v <= hi_f)]
    return {"q1": q1, "median": med, "q3": q3, "mean": float(v.mean()),
            "lowerfence": float(inside.min()), "upperfence": float(inside.max())}


def _thin(frame: pd.DataFrame, cap: int = MAX_POINTS) -> pd.DataFrame:
    """Every outlier beyond 1.5 IQR plus a seeded sample of at most ``cap`` other cases."""
    if len(frame) <= cap:
        return frame
    st = _box_stats(frame["value"])
    out = (frame["value"] < st["lowerfence"]) | (frame["value"] > st["upperfence"])
    rest = frame[~out].sample(n=min(cap, int((~out).sum())), random_state=0)
    return pd.concat([frame[out], rest]).sort_index()


def group_iqm_div(per_call_df=None, predictions_df=None) -> str | None:
    """Horizontal box+strip of every metric at once — the group IQM overview."""
    if not has_plotly():
        return None
    series = _collect_series(per_call_df, predictions_df)
    if not series:
        return None
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    directions = _directions()
    metrics = sorted(series)
    # One row and one x-axis per metric: PSNR (~20 dB) and MSE (~1e-3) on a shared axis
    # left every small-valued metric as a smear at zero.
    fig = make_subplots(rows=len(metrics), cols=1, vertical_spacing=min(0.04, 0.6 / len(metrics)))
    thinned = False
    for row, m in enumerate(metrics, start=1):
        frame = series[m]
        name = _arrow(m, directions)
        st = _box_stats(frame["value"])
        fig.add_trace(
            go.Box(
                y=[name], orientation="h", **{k: [v] for k, v in st.items()},
                boxpoints=False, line={"width": 1.2, "color": role("series")},
                fillcolor="rgba(31,95,153,0.10)", hoverinfo="skip",
            ),
            row=row, col=1,
        )
        pts = _thin(frame)
        thinned |= len(pts) < len(frame)
        # Dots ride an invisible box on the same row, so plotly jitters them for us.
        fig.add_trace(
            go.Box(
                x=pts["value"], y=[name] * len(pts), orientation="h", boxpoints="all",
                jitter=0.5, pointpos=0, line={"width": 0, "color": "rgba(0,0,0,0)"},
                fillcolor="rgba(0,0,0,0)", whiskerwidth=0,
                marker={"size": 4, "color": role("series"), "line": {"color": _SURFACE_LINE, "width": 0.4}},
                text=pts["label"], hovertemplate="%{text}: %{x:.4g}<extra>" + m + "</extra>",
            ),
            row=row, col=1,
        )
    apply_theme(fig, title="Group IQM distribution", height=72 * len(metrics) + 110)
    fig.update_layout(showlegend=False, boxmode="overlay")
    fig.update_xaxes(nticks=5, tickfont={"size": 10})
    fig.update_yaxes(automargin=True, showgrid=False)
    seat_menus(fig)
    body = to_div(fig, div_id="iqm-group")
    return body + caption(
        "One row per metric across every recorded case, each on its own scale. The box spans the "
        "inter-quartile range, the dashed mean marker is the group average, and "
        "each dot is one case (hover for its id). Points beyond the whiskers are "
        "1.5x-IQR outliers — the cases worth inspecting in the mosaic below. "
        "Arrows show whether higher or lower is better."
        + (f" Boxes use every case; dots show every outlier and a random {MAX_POINTS} of the rest."
           if thinned else "")
    )


def learning_curve_div(df) -> str | None:
    """Train/val curves with a metric dropdown and a log-y toggle."""
    if not has_plotly() or df is None or getattr(df, "empty", True):
        return None
    need = {"split", "step", "metric", "value"}
    if not need.issubset(df.columns):
        return None
    curve = df[df["split"].isin(["train", "val"])].copy()
    if curve.empty:
        return None
    import plotly.graph_objects as go

    method_col = "method" if "method" in curve.columns else "run_id"
    curve["metric"] = curve["metric"].astype(str)
    # The ``val_`` prefix on validation metrics encodes the split, which the
    # ``split`` column already carries. Strip it so a validation series pairs with
    # its training twin on one panel (solid = train, dotted = val) instead of
    # appearing as an unpaired, differently-named dropdown entry.
    curve["base_metric"] = curve["metric"].str.replace(r"^val_", "", regex=True)
    metrics = sorted(curve["base_metric"].unique())
    # Default to a populated loss panel (never the hardcoded "loss" when it is absent).
    default = next((m for m in ("loss", "g_total_loss") if m in metrics), metrics[0])
    methods = sorted(curve[method_col].astype(str).unique()) if method_col in curve else ["model"]
    colour = arm_palette(methods)

    fig = go.Figure()
    trace_metric: list[str] = []
    for metric in metrics:
        for mth in methods:
            for split in ("train", "val"):
                sel = curve[
                    (curve["base_metric"] == metric)
                    & (curve[method_col].astype(str) == mth)
                    & (curve["split"] == split)
                ].sort_values("step")
                if sel.empty:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=sel["step"],
                        y=sel["value"],
                        mode="lines",
                        name=f"{mth} · {split}",
                        line={
                            "color": colour[mth],
                            "width": 2,
                            "dash": "solid" if split == "train" else "dot",
                        },
                        visible=(metric == default),
                        hovertemplate="step %{x}: %{y:.4g}<extra>"
                        + f"{mth} {split} {metric}</extra>",
                    )
                )
                trace_metric.append(metric)
    if not trace_metric:
        return None

    buttons = []
    for metric in metrics:
        vis = [tm == metric for tm in trace_metric]
        buttons.append(
            {
                "label": metric,
                "method": "update",
                "args": [{"visible": vis}, {"yaxis.title.text": metric}],
            }
        )
    apply_theme(fig, title="Training dynamics", height=460)
    fig.update_layout(
        xaxis_title="step",
        yaxis_title=default,
        legend={"orientation": "h", "y": -0.18},
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.0,
                "xanchor": "left",
            },
            {
                "type": "buttons",
                "direction": "right",
                "showactive": False,
                "x": 1.0,
                "xanchor": "right",
                "buttons": [
                    {"label": "linear-y", "method": "relayout", "args": [{"yaxis.type": "linear"}]},
                    {"label": "log-y", "method": "relayout", "args": [{"yaxis.type": "log"}]},
                ],
            },
        ],
    )
    seat_menus(fig)
    body = to_div(fig, div_id="learning-curves")
    return body + caption(
        "Each line is a metric vs training step; solid = train, dotted = "
        "validation, colour = method. Pick the metric from the dropdown and "
        "toggle log-y for loss curves. A widening train/val gap signals "
        "over-fitting; click legend entries to isolate a method."
    )


def metric_distribution_div(per_call_df=None, predictions_df=None) -> str | None:
    """Per-metric distribution (violin+box+points) with a metric dropdown."""
    if not has_plotly():
        return None
    import plotly.graph_objects as go

    # Prefer the long predictions frame (carries method) so distributions can be
    # split by method; else the wide per-case frame (single method).
    by_method: dict[str, dict[str, pd.DataFrame]] = {}
    if (
        predictions_df is not None
        and not predictions_df.empty
        and {"metric", "value"}.issubset(predictions_df.columns)
    ):
        mcol = "method" if "method" in predictions_df.columns else None
        for mth, grp in predictions_df.groupby(mcol) if mcol else [("model", predictions_df)]:
            by_method[str(mth)] = _series_from_long(grp)
    if not by_method:
        series = _collect_series(per_call_df, None)
        if series:
            by_method = {"model": series}
    if not by_method:
        return None

    metrics = sorted({m for s in by_method.values() for m in s})
    if not metrics:
        return None
    methods = sorted(by_method)
    colour = arm_palette(methods)
    default = "psnr" if "psnr" in metrics else metrics[0]

    fig = go.Figure()
    thinned = False
    trace_metric: list[str] = []
    for metric in metrics:
        for mth in methods:
            frame = by_method[mth].get(metric)
            if frame is None or frame.empty:
                continue
            full_n = len(frame)
            frame = _thin(frame)
            thinned |= len(frame) < full_n
            fig.add_trace(
                go.Violin(
                    y=frame["value"],
                    name=mth,
                    legendgroup=mth,
                    box_visible=True,
                    meanline_visible=True,
                    points="all",
                    jitter=0.4,
                    pointpos=0,
                    line_color=colour[mth],
                    marker={"size": 4},
                    text=frame["label"],
                    hovertemplate="%{text}: %{y:.4g}<extra>" + f"{mth} {metric}</extra>",
                    visible=(metric == default),
                )
            )
            trace_metric.append(metric)
    if not trace_metric:
        return None

    buttons = [
        {
            "label": metric,
            "method": "update",
            "args": [
                {"visible": [tm == metric for tm in trace_metric]},
                {"yaxis.title.text": metric},
            ],
        }
        for metric in metrics
    ]
    apply_theme(fig, title="Per-case metric distribution", height=440)
    fig.update_layout(
        yaxis_title=default,
        violinmode="group",
        showlegend=len(methods) > 1,
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.0,
                "xanchor": "left",
            }
        ],
    )
    seat_menus(fig)
    body = to_div(fig, div_id="metric-dist")
    return body + caption(
        "The distribution of a single metric across cases: the violin is the "
        "kernel density, the inner box the IQR, and every dot a case (hover for "
        "its id). When several methods are present they sit side by side so you "
        "can compare spread, not just the mean. Choose the metric from the "
        "dropdown."
        + (f" With many cases the shape is drawn from every outlier and a random {MAX_POINTS} "
           "of the rest." if thinned else "")
    )


def comparison_div(predictions_df=None) -> str | None:
    """Bland-Altman agreement between the two best-represented methods."""
    if not has_plotly() or predictions_df is None or predictions_df.empty:
        return None
    cols = {"method", "metric", "value", "subject_id"}
    if not cols.issubset(predictions_df.columns):
        return None
    methods = predictions_df["method"].astype(str).value_counts()
    if len(methods) < 2:
        return None
    import plotly.graph_objects as go

    m_a, m_b = methods.index[0], methods.index[1]
    metrics = sorted(predictions_df["metric"].astype(str).unique())
    fig = go.Figure()
    trace_metric: list[str] = []
    default = metrics[0]
    for metric in metrics:
        sub = predictions_df[predictions_df["metric"].astype(str) == metric]
        wide = sub.pivot_table(index="subject_id", columns="method", values="value", aggfunc="mean")
        if m_a not in wide.columns or m_b not in wide.columns:
            continue
        pair = wide[[m_a, m_b]].dropna()
        if len(pair) < 2:
            continue
        mean = (pair[m_a] + pair[m_b]) / 2.0
        diff = pair[m_a] - pair[m_b]
        bias = float(diff.mean())
        sd = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
        vis = metric == default
        fig.add_trace(
            go.Scatter(
                x=mean,
                y=diff,
                mode="markers",
                name=metric,
                marker={"size": 8, "color": role("series")},
                text=[str(i) for i in pair.index],
                visible=vis,
                hovertemplate="%{text}<br>mean %{x:.4g}, diff %{y:.4g}<extra></extra>",
            )
        )
        for val, lab, colr in (
            (bias, "bias", flag_colour()),
            (bias + 1.96 * sd, "+1.96 SD", _MUTED_LINE),
            (bias - 1.96 * sd, "-1.96 SD", _MUTED_LINE),
        ):
            fig.add_trace(
                go.Scatter(
                    x=[float(mean.min()), float(mean.max())],
                    y=[val, val],
                    mode="lines",
                    name=lab,
                    line={"color": colr, "dash": "dash", "width": 1},
                    showlegend=False,
                    visible=vis,
                    hoverinfo="skip",
                )
            )
        trace_metric += [metric] * 4
    if not trace_metric:
        return None

    buttons = [
        {
            "label": metric,
            "method": "update",
            "args": [{"visible": [tm == metric for tm in trace_metric]}],
        }
        for metric in sorted(set(trace_metric))
    ]
    apply_theme(fig, title=f"Bland-Altman: {m_a} vs {m_b}", height=440)
    fig.update_layout(
        xaxis_title="mean of the two methods",
        yaxis_title=f"{m_a} - {m_b}",
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.0,
                "xanchor": "left",
            }
        ],
    )
    seat_menus(fig)
    body = to_div(fig, div_id="bland-altman")
    return body + caption(
        f"Agreement between {m_a} and {m_b} per subject: x is their average, y "
        "their difference. The red line is the mean bias and the dashed "
        "lines the +/-1.96 SD limits of agreement; points outside them are the "
        "subjects where the two methods disagree most. Pick the metric from the "
        "dropdown."
    )
