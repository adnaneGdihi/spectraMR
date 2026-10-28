r"""Row selection and header rendering for the headline results table."""

from __future__ import annotations

import pandas as pd

#: Identity (zero-filled) scores a recon table must sit beside: a model that
#: does not beat ``zf_psnr`` has learned nothing the input did not already hold.
_IDENTITY = {"zf_psnr", "zf_ssim"}


def _headline_rows(df: pd.DataFrame, view: str) -> pd.DataFrame:
    """One value per (method, base metric): the test split, else the ``view`` pass.

    Logged names carry a split prefix and per-acceleration suffixes
    (``val_psnr_8x``); matching the bare headline set against them selected
    nothing, so the table was never written. Rows are re-keyed on
    ``base_metric`` and only headline columns (plain or cascade ``_mean``) are
    kept. A plain column wins over a ``_mean`` of the same base.
    """
    from spectramr.infrastructure.reporting.metric_names import annotate_frame
    from spectramr.infrastructure.reporting.views import snapshot

    if "base_metric" not in df.columns:
        df = annotate_frame(df)
    if "method" not in df.columns:
        df = df.assign(method="run")
    # Per method: its test split when it has one, else its ``view`` pass. Choosing
    # once for the whole frame dropped every method without a test split.
    if "split" in df.columns:
        tested = set(df.loc[df["split"] == "test", "method"])
        parts = [df[(df["split"] == "test") & df["method"].isin(tested)]]
        rest = df[~df["method"].isin(tested)]
        if not rest.empty:
            parts.append(snapshot(rest, view))
        rows = (
            pd.concat([p for p in parts if not p.empty], ignore_index=True)
            if any(not p.empty for p in parts)
            else df.iloc[0:0]
        )
    else:
        rows = df
    if rows.empty:
        return rows
    rows = rows[rows["headline"].astype(bool)].copy()
    is_mean = rows["metric"].astype(str).str.endswith("_mean")
    plain = set(zip(rows.loc[~is_mean, "method"], rows.loc[~is_mean, "base_metric"], strict=True))
    shadowed = is_mean & pd.Series(
        [(m, b) in plain for m, b in zip(rows["method"], rows["base_metric"], strict=True)],
        index=rows.index,
    )
    rows = rows[~shadowed]
    rows["metric"] = rows["base_metric"]
    return rows


def spread_note(rows: pd.DataFrame, view: str, any_test: bool) -> str:
    """Caption sentence naming what each mean is over, or that there is no spread."""
    per = rows.groupby(["method", "metric"]).size()
    if per.empty or per.max() < 2:
        where = "on the test split" if any_test else f"at its {view} validation pass"
        return f"One value per method, read {where}; no spread is reported."
    if "subject_id" in rows.columns and rows["subject_id"].notna().any():
        return r"Mean $\pm$ std over subjects."
    if "run_id" in rows.columns and rows.groupby("method")["run_id"].nunique().max() > 1:
        return r"Mean $\pm$ std over runs."
    return r"Mean $\pm$ std over repeated rows."


def _header(metric: str, higher: bool | None) -> str:
    """Readable column header with a direction arrow (``PSNR (dB) ↑``)."""
    from spectramr.infrastructure.reporting.style import pretty_label

    label = "zero-filled " + pretty_label(metric[3:]) if metric.startswith("zf_") else None
    label = label or pretty_label(metric)
    return label if higher is None else f"{label} {'↑' if higher else '↓'}"


def _tex(text: str) -> str:
    """Escape the characters a metric or arm name can carry into LaTeX."""
    for raw, esc in (
        ("\\", r"\textbackslash{}"),
        ("_", r"\_"),
        ("%", r"\%"),
        ("&", r"\&"),
        ("#", r"\#"),
        ("↑", r"$\uparrow$"),
        ("↓", r"$\downarrow$"),
    ):
        text = text.replace(raw, esc)
    return text
