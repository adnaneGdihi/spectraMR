r"""Default panel set for the learning-curves figure.

A learning-curve figure answers three questions -- is it converging, is it
generalising, and is it better than doing nothing -- so the default is one panel
per quantity with train and validation *on the same axes*:

* ``loss``: the training total (one column; ``loss`` and ``g_total_loss`` are the
  same series under two names on most strategies) against the validation loss.
* ``psnr`` / ``ssim``: training against validation, plus the zero-filled
  identity score (``val_zf_*``) as a reference line -- a model under that line has
  learned nothing the input did not already hold.

Picking "every metric whose name contains loss" instead gave six panels, two of
them the same series and three empty (``best``/``final`` scalars have no step).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from spectramr.infrastructure.reporting.metric_names import annotate_frame

#: Training-total column names, in preference order.
TRAIN_TOTALS: tuple[str, ...] = ("loss", "total_loss", "g_total_loss", "generator_loss")


@dataclass(frozen=True)
class Panel:
    """One axes: a quantity, its train/val rows and an optional identity row."""

    title: str
    train: pd.DataFrame
    val: pd.DataFrame
    identity: pd.DataFrame


def ema(values: np.ndarray, half_life: float = 8.0) -> np.ndarray:
    """Exponential moving average with the given half-life in samples; NaN-tolerant."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values.copy()
    alpha = 1.0 - np.exp(-np.log(2.0) / max(half_life, 1.0))
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, values.size):
        out[i] = out[i - 1] if np.isnan(values[i]) else alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def _stepped(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["step"].fillna(-1) >= 0) & df["value"].notna()]


def _pick(df: pd.DataFrame, split: str, bases: tuple[str, ...]) -> pd.DataFrame:
    """Per method, rows of its first base in ``bases`` in ``split``; plain over ``_mean``.

    Chosen per method: arms that log ``loss`` beside arms that log
    ``g_total_loss``, or ``val_psnr`` beside a cascade's ``val_psnr_mean``,
    must all reach the panel.
    """
    rows = df[(df["split"] == split) & df["headline"].astype(bool)]
    picked = []
    for _, own in rows.groupby("method", sort=False):
        for base in bases:
            hit = own[own["base_metric"] == base]
            if hit.empty:
                continue
            plain = hit[~hit["metric"].astype(str).str.endswith("_mean")]
            picked.append(plain if not plain.empty else hit)
            break
    return pd.concat(picked) if picked else rows.iloc[0:0]


def default_panels(df: pd.DataFrame) -> list[Panel]:
    """The panels the run has data for, in loss / PSNR / SSIM order."""
    if df is None or df.empty or not {"metric", "split", "step", "value"} <= set(df.columns):
        return []
    if "base_metric" not in df.columns:
        df = annotate_frame(df)
    df = _stepped(df)
    if df.empty:
        return []
    panels = []
    for title, bases in (("loss", TRAIN_TOTALS), ("psnr", ("psnr",)), ("ssim", ("ssim",))):
        val_bases = ("loss",) if title == "loss" else bases
        train, val = _pick(df, "train", bases), _pick(df, "val", val_bases)
        identity = _pick(df, "val", (f"zf_{title}",)) if title != "loss" else df.iloc[0:0]
        if not train.empty or not val.empty:
            panels.append(Panel(title, train, val, identity))
    return panels


__all__ = ["TRAIN_TOTALS", "Panel", "default_panels", "ema"]
