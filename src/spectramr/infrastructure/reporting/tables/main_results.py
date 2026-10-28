r"""Table 2.1 — Main results table.

Rows = methods, columns = metrics; mean ± std over seeds; best in
**bold**, second-best *underlined*; significance superscript (e.g. ``*``
for p < 0.05 vs. baseline after Holm correction). Includes a parameter /
FLOPs cost column where present.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ._direction import direction
from ._headline import _IDENTITY, _header, _headline_rows, _tex, spread_note
from ._significance import paired_significance

#: Validation readings of the disentangled strategy (base names, split prefix stripped):
#: what a comparison of content/style arms is about, shown whenever an arm reports them.
#: The chance level and the probe-split flag are references, not results, and stay out.
_DISENTANGLEMENT = {
    "style_contrast_acc",
    "content_contrast_leak",
    "swap_content_distance",
    "swap_style_distance",
}


def _format_value(mean: float, std: float | None, decimals: int) -> str:
    if std is None or np.isnan(std):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def _sup(text: str) -> str:
    return text.replace("‡", r"$^\ddagger$").replace("§", r"$^\S$")


def make(
    df: pd.DataFrame,
    out_dir: str | Path,
    *,
    metrics: list[str] | None = None,
    cost_metrics: list[str] = ["params_m"],
    higher_is_better: dict[str, bool] | None = None,
    baseline_method: str | None = None,
    decimals: int = 3,
    view: str = "selected",
    run_notes: dict[str, str] | None = None,
    excluded_runs: dict[str, str] | None = None,
    arm_notes: dict[str, str] | None = None,
) -> dict[str, Path] | None:
    """Build the main results table in Markdown + LaTeX.

    Args:
        metrics: List of metric names to include. Defaults to common test
            metrics in the test split.
        cost_metrics: Cost columns appended to the right (e.g. ``params_m``).
        higher_is_better: Per-metric direction of "better" (default True
            for everything except metrics whose name contains ``loss``,
            ``error``, ``rmse``, ``nmse``, ``mae``, ``fid``, ``kid``,
            ``hd95``, ``ece``, ``nll``).
        baseline_method: If supplied, run paired t-test of every other
            method against the baseline and Holm-correct.
        view: Checkpoint view (:mod:`..views`) used when the run has no
            ``test`` split.
        run_notes: ``{method: note}`` for tabulated runs whose completion is
            not confirmed (:mod:`..run_status`); each is marked ‡ and its note
            printed under the table.
        excluded_runs: ``{method: note}`` for runs left out of the table,
            e.g. a crashed arm; named under the table and in the caption.
        arm_notes: ``{method: note}`` from each arm's ``metadata.comparison_note``;
            each is marked § and its note printed under the table and in the caption.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if df.empty:
        return None
    test = _headline_rows(df, view)
    if "split" in df.columns and cost_metrics:
        # cost facts (params_m, ...) live in the run split, outside any pass
        cost = df[(df["split"] == "run") & df["metric"].isin(cost_metrics)]
        test = pd.concat([test, cost], ignore_index=True) if not cost.empty else test

    if metrics is None:
        common = {
            "psnr",
            "ssim",
            "ms_ssim",
            "lpips",
            "vif",
            "fsim",
            "haarpsi",
            "rmse",
            "nmse",
            "nrmse",
            "hfen",
            "gmsd",
            "fid",
            "kid",
            "dice",
            "hd95",
            "auroc",
            "ece",
        }
        metrics = sorted(set(test["metric"].unique()) & (common | _IDENTITY | _DISENTANGLEMENT))
    if not metrics:
        return None

    higher_is_better = higher_is_better or {}

    def hib(m: str) -> bool | None:
        return direction(m, higher_is_better)

    methods = sorted(test["method"].unique())
    tested_methods = set(test.loc[test["split"] == "test", "method"]) if "split" in test else set()
    mixed_sources = bool(tested_methods) and tested_methods != set(methods)
    rows: list[dict] = []
    for m in methods:
        row: dict = {"method": m}
        for metric in metrics + cost_metrics:
            sel = test[(test["method"] == m) & (test["metric"] == metric)].dropna(subset=["value"])
            # keyed by subject when the rows carry one, so a paired test pairs subjects
            key = sel["subject_id"].astype(str) if "subject_id" in sel.columns else None
            vals = pd.Series(
                sel["value"].astype(float).to_numpy(),
                index=key.to_numpy() if key is not None and key.notna().all() else None,
            )
            if vals.empty:
                row[metric] = (np.nan, np.nan, np.nan)
            else:
                std = vals.std(ddof=1) if vals.size > 1 else np.nan
                row[metric] = (vals.mean(), std, vals)
        rows.append(row)

    # Find best and second-best per metric column
    best_idx: dict[str, int] = {}
    second_idx: dict[str, int] = {}
    for metric in metrics:
        means = [r[metric][0] for r in rows]
        valid = [(i, v) for i, v in enumerate(means) if not np.isnan(v)]
        # Ranking one entry, a tie, the shared zero-filled reference, or values from
        # two sources (test split beside a validation pass) is not a result.
        if (
            len(valid) < 2
            or metric.startswith("zf_")
            or len({v for _, v in valid}) == 1
            or mixed_sources
            or hib(metric) is None  # no declared direction: shown, never ranked
        ):
            continue
        valid.sort(key=lambda iv: iv[1], reverse=bool(hib(metric)))
        best_idx[metric] = valid[0][0]
        if len(valid) > 1:
            second_idx[metric] = valid[1][0]

    sig = paired_significance(rows, methods, metrics, baseline_method)

    # A cost column nobody measured is dropped, not printed as a column of dashes.
    cost_metrics = [c for c in cost_metrics if any(not np.isnan(r[c][0]) for r in rows)]
    labels = [_header(m, hib(m)) for m in metrics] + [_header(c, None) for c in cost_metrics]

    # Render Markdown
    cols = ["Method", *labels]
    md_lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    notes = {m: n for m, n in (run_notes or {}).items() if m in methods}
    arm = {m: n for m, n in (arm_notes or {}).items() if m in methods}

    def _name(m: str) -> str:
        name = f"{m} †" if mixed_sources and m not in tested_methods else m
        name = f"{name} ‡" if m in notes else name
        return f"{name} §" if m in arm else name

    for i, r in enumerate(rows):
        cells = [_name(str(r["method"]))]
        for metric in metrics:
            mean, std, _ = r[metric]
            text = _format_value(mean, std, decimals)
            if i == best_idx.get(metric):
                text = f"**{text}**"
            elif i == second_idx.get(metric):
                text = f"_{text}_"
            if sig.get((r["method"], metric)):
                text += "*"
            cells.append(text)
        for cm in cost_metrics:
            mean, _, _ = r[cm]
            cells.append(_format_value(mean, None, 1) if not np.isnan(mean) else "—")
        md_lines.append("| " + " | ".join(cells) + " |")
    footnotes = [f"‡ {m}: {n}." for m, n in notes.items()]
    footnotes += [f"§ {m}: {n}." for m, n in arm.items()]
    footnotes += [f"Not tabulated: {m} ({n})." for m, n in (excluded_runs or {}).items()]
    md_path = out_dir / "tab_2_1_main_results.md"
    md_path.write_text("\n".join([*md_lines, *(["", *footnotes] if footnotes else [])]))

    # Render LaTeX (IEEE style). The caption names the spread only when one exists.
    any_test = "split" in df.columns and (df["split"] == "test").any()
    spread = spread_note(test[test["metric"].isin(metrics)], view, any_test)
    if mixed_sources:
        spread += (
            r" $^\dagger$ read at the " + view + " validation pass, the others on the test split;"
            " the two are not ranked against each other."
        )
    spread += "".join(" " + _sup(_tex(f)) for f in footnotes)
    tested = (
        r" * $p<0.05$ vs. " + _tex(str(baseline_method)) + " (paired t-test, Holm-corrected)."
        if sig
        else ""
    )
    tex_lines = [
        r"\begin{table}[t]\centering",
        r"\caption{Main results. " + spread + r" Best in \textbf{bold},"
        r" second-best in \emph{italics}." + tested + "}",
        r"\label{tab:main_results}",
        r"\begin{tabular}{l" + "c" * (len(metrics) + len(cost_metrics)) + "}",
        r"\toprule",
        " & ".join(_tex(c) for c in cols) + r" \\",
        r"\midrule",
    ]
    for i, r in enumerate(rows):
        name = _tex(_name(str(r["method"])))
        cells = [_sup(name.replace("†", r"$^\dagger$"))]
        for metric in metrics:
            mean, std, _ = r[metric]
            text = _format_value(mean, std, decimals)
            if i == best_idx.get(metric):
                text = r"\textbf{" + text + "}"
            elif i == second_idx.get(metric):
                text = r"\emph{" + text + "}"
            if sig.get((r["method"], metric)):
                text += r"$^{*}$"
            cells.append(text)
        for cm in cost_metrics:
            mean, _, _ = r[cm]
            cells.append(_format_value(mean, None, 1) if not np.isnan(mean) else "—")
        tex_lines.append(" & ".join(cells) + r" \\")
    tex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex_path = out_dir / "tab_2_1_main_results.tex"
    tex_path.write_text("\n".join(tex_lines))

    # Machine-readable CSV sink: raw mean/std/cost values (no markup) so
    # downstream analysis doesn't have to re-parse the formatted LaTeX.
    # Function-local import avoids the tables/__init__ circular import
    # (the package imports this submodule before emit_table_csv is defined).
    from spectramr.infrastructure.reporting.tables import emit_table_csv

    flat: list[dict] = []
    for r in rows:
        rec: dict = {"method": r["method"]}
        for metric in metrics:
            mean, std, _ = r[metric]
            rec[f"{metric}_mean"] = mean
            rec[f"{metric}_std"] = std
        for cm in cost_metrics:
            rec[cm] = r[cm][0]
        if notes:
            rec["note"] = notes.get(r["method"], "")
        if arm:
            rec["arm_note"] = arm.get(r["method"], "")
        flat.append(rec)
    csv_path = emit_table_csv(pd.DataFrame(flat), out_dir, "tab_2_1_main_results")

    return {"markdown": md_path, "latex": tex_path, "csv": csv_path}
