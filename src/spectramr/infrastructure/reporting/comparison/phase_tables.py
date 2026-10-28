r"""One value per arm, metric and phase, as data a paper can quote.

* ``val``: the metric at the pass each arm is read at (:class:`.arms.Arms` steps).
* ``train``: the mean of the last ``window`` logged rows at or before that pass.
  ``training_metrics.csv`` is appended across resumes, so a step logged twice keeps
  its last row. Training writes no per-item rows, so this is descriptive only.
* ``test``: the mean over the cases of ``final_eval.json``, with their spread.

``inference_metrics.csv`` records why a metric is absent from an inference run (a
full-reference metric on a run given no reference, say); that reason is returned
beside the table instead of being lost.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from spectramr.core.metrics.metric_directions import resolve_direction

from ..per_slice import SlicePass, find_rows_files, read_pass
from ..views import snapshot
from .arms import Arms
from .metric_selection import PHASES, check_phases, frame_of

logger = logging.getLogger(__name__)

COLUMNS: tuple[str, ...] = (
    "arm",
    "phase",
    "metric",
    "mean",
    "std",
    "n",
    "unit",
    "step",
    "direction",
)


PHASE_LABELS = {"val": "validation", "train": "training", "test": "inference"}


def direction_word(metric: str) -> str:
    """``higher`` / ``lower`` from the direction SSOT; empty when it declares none."""
    higher = resolve_direction(metric)
    return "" if higher is None else ("higher" if higher else "lower")


def _row(arm, phase, metric, values, unit, step) -> dict:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return {
        "arm": arm,
        "phase": phase,
        "metric": metric,
        "mean": float(v.mean()) if v.size else np.nan,
        "std": float(v.std(ddof=1)) if v.size > 1 else np.nan,
        "n": int(v.size),
        "unit": unit,
        "step": step,
        "direction": direction_word(metric),
    }


def _slice_pass(arms: Arms, arm: str) -> SlicePass | None:
    """The arm's per-slice rows at its read pass, or ``None`` when it wrote none usable."""
    files = find_rows_files(arms.run_dirs[arm])
    if not files:
        return None
    try:
        return read_pass(files, arms.steps.get(arm))
    except ValueError as exc:  # rows that cannot be paired (batch means, no identity)
        logger.warning("%s: no per-slice spread for the validation pass: %s", arm, exc)
        return None


def _slice_spread(rows: SlicePass | None, metric: str) -> tuple[float, int] | None:
    """SD and count of the finite per-slice values of ``metric``, or ``None`` without a column."""
    column = next((c for c in (f"val_{metric}", metric) if rows and c in rows.metric_columns), None)
    if column is None:
        return None
    values = pd.to_numeric(rows.frame[column], errors="coerce").to_numpy(float)
    values = values[np.isfinite(values)]
    return (float(values.std(ddof=1)), int(values.size)) if values.size > 1 else None


def _val_rows(arms: Arms, metrics: list[str]) -> list[dict]:
    passes = snapshot(arms.df, arms.view)
    rows = []
    for arm in arms.labels:
        mine = passes[(passes["method"] == arm) & passes["headline"]]
        slices = _slice_pass(arms, arm)
        for metric in metrics:
            vals = mine.loc[mine["base_metric"] == metric, "value"]
            if vals.empty:
                continue
            row = _row(arm, "val", metric, [vals.iloc[-1]], "validation pass", arms.steps.get(arm))
            # The pass value stays the mean (it is what the arm was selected on); the slices
            # behind it supply the spread and n a bar needs.
            if (spread := _slice_spread(slices, metric)) is not None:
                row.update(std=spread[0], n=spread[1], unit="validation slices (slice SD)")
            rows.append(row)
    return rows


def _train_rows(arms: Arms, metrics: list[str], window: int) -> list[dict]:
    rows = []
    train = frame_of(arms, "train")
    for arm in arms.labels:
        mine = train[train["method"] == arm]
        limit = arms.steps.get(arm)
        for metric in metrics:
            one = mine[mine["base_metric"] == metric].sort_values("step", kind="stable")
            one = one.drop_duplicates("step", keep="last")
            if limit is not None:
                one = one[one["step"] <= limit]
            tail = one.tail(window)
            if not tail.empty:
                unit = f"last {len(tail)} logged training rows"
                rows.append(
                    _row(arm, "train", metric, tail["value"], unit, int(tail["step"].max()))
                )
    return rows


def _test_rows(arms: Arms, metrics: list[str]) -> list[dict]:
    rows = []
    test = frame_of(arms, "test")
    for arm in arms.labels:
        mine = test[test["method"] == arm]
        for metric in metrics:
            vals = mine.loc[mine["base_metric"] == metric, "value"]
            if not vals.empty:
                rows.append(_row(arm, "test", metric, vals, "inference cases", None))
    return rows


def phase_table(
    arms: Arms,
    metrics_by_phase: dict[str, list[str]],
    *,
    train_window: int = 50,
) -> pd.DataFrame:
    """Long table (:data:`COLUMNS`) of the requested metrics in each requested phase."""
    if train_window < 1:
        raise ValueError(f"train_window must be at least 1, got {train_window}")
    builders = {
        "val": lambda m: _val_rows(arms, m),
        "train": lambda m: _train_rows(arms, m, train_window),
        "test": lambda m: _test_rows(arms, m),
    }
    rows: list[dict] = []
    for phase in check_phases(list(metrics_by_phase)):
        rows += builders[phase](metrics_by_phase[phase])
    return pd.DataFrame(rows, columns=list(COLUMNS))


def wide(table: pd.DataFrame, phase: str) -> pd.DataFrame:
    """One row per arm, one column per metric: the phase's means."""
    one = table[table["phase"] == phase]
    return one.pivot(index="arm", columns="metric", values="mean").reset_index()


def latex_table(table: pd.DataFrame, phase: str) -> str:
    """A booktabs table of one phase: arms down, metrics across, best in bold.

    Best is marked only for a metric whose direction the SSOT declares and whose
    arms do not tie; spread is printed where the phase has one.
    """
    from ..tables._headline import _header, _tex

    one = table[table["phase"] == phase]
    metrics = list(dict.fromkeys(one["metric"]))
    arms = list(dict.fromkeys(one["arm"]))
    best: dict[str, float] = {}
    for m in metrics:
        col = one[one["metric"] == m]
        word, means = col["direction"].iloc[0], col["mean"].dropna()
        if word and means.nunique() > 1:
            best[m] = means.max() if word == "higher" else means.min()
    head = ["arm"] + [_tex(_header(m, resolve_direction(m))) for m in metrics]
    rows = []
    for arm in arms:
        cells = [_tex(arm)]
        for m in metrics:
            cell = one[(one["arm"] == arm) & (one["metric"] == m)]
            if cell.empty or pd.isna(cell["mean"].iloc[0]):
                cells.append("--")
                continue
            mean, std = float(cell["mean"].iloc[0]), cell["std"].iloc[0]
            text = f"{mean:.4g}" + ("" if pd.isna(std) else rf" $\pm$ {std:.2g}")
            cells.append(rf"\textbf{{{text}}}" if best.get(m) == mean else text)
        rows.append(" & ".join(cells) + r" \\")
    unit = "; ".join(sorted(set(one["unit"])))
    return "\n".join(
        [
            r"\begin{tabular}{l" + "c" * len(metrics) + "}",
            r"\toprule",
            " & ".join(head) + r" \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            f"% {PHASE_LABELS[phase]}: {unit}",
        ]
    )


def inference_unavailable(run_dir: str | Path, arm: str) -> list[dict[str, str]]:
    """Metrics an inference run declared but did not compute, with the recorded reason."""
    path = Path(run_dir) / "inference_metrics.csv"
    if not path.is_file():
        return []
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not {"metric", "status", "reason"} <= set(frame.columns):
        return []
    skipped = frame[frame["status"].str.strip().ne("computed") & frame["status"].str.strip().ne("")]
    return [
        {"arm": arm, "phase": "test", "metric": r.metric, "reason": f"{r.status}: {r.reason}"}
        for r in skipped.drop_duplicates("metric").itertuples()
    ]


__all__ = ["COLUMNS", "PHASES", "direction_word", "inference_unavailable", "phase_table", "wide"]
