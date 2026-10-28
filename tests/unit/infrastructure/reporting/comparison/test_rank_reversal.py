"""``comparison/rank_reversal.py``: ranks follow each metric's declared direction."""

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from spectramr.infrastructure.reporting.comparison.rank_reversal import make, rank_table


def _table(rows):
    """``rows``: ``(arm, metric, mean, direction)`` in the validation phase."""
    return pd.DataFrame(
        [
            {
                "arm": a,
                "phase": "val",
                "metric": m,
                "mean": v,
                "std": float("nan"),
                "n": 1,
                "unit": "x",
                "step": 1,
                "direction": d,
            }
            for a, m, v, d in rows
        ]
    )


def test_a_reversal_between_two_metrics_shows_as_crossing_ranks():
    table = _table(
        [
            ("a", "psnr", 30.0, "higher"),
            ("b", "psnr", 28.0, "higher"),
            ("a", "nmse", 0.09, "lower"),
            ("b", "nmse", 0.05, "lower"),
        ]
    )
    ranks = rank_table(table, "val")
    assert ranks.loc["a"].to_dict() == {"nmse": 2, "psnr": 1}
    assert ranks.loc["b"].to_dict() == {"nmse": 1, "psnr": 2}


def test_a_metric_with_no_direction_is_not_ranked():
    table = _table(
        [
            ("a", "psnr", 30.0, "higher"),
            ("b", "psnr", 28.0, "higher"),
            ("a", "mystery", 1.0, ""),
            ("b", "mystery", 2.0, ""),
        ]
    )
    assert list(rank_table(table, "val").columns) == ["psnr"]


def test_ties_share_the_smaller_rank():
    table = _table(
        [
            ("a", "psnr", 30.0, "higher"),
            ("b", "psnr", 30.0, "higher"),
            ("c", "psnr", 25.0, "higher"),
        ]
    )
    assert rank_table(table, "val")["psnr"].tolist() == [1, 1, 3]


def test_the_figure_needs_two_ranked_metrics(tmp_path):
    one = _table([("a", "psnr", 30.0, "higher"), ("b", "psnr", 28.0, "higher")])
    assert make(pd.DataFrame(), tmp_path / "r.png", phase_df=one, arms=["a", "b"]) is None
    two = _table(
        [
            ("a", "psnr", 30.0, "higher"),
            ("b", "psnr", 28.0, "higher"),
            ("a", "nmse", 0.09, "lower"),
            ("b", "nmse", 0.05, "lower"),
            ("a", "mystery", 1.0, ""),
            ("b", "mystery", 2.0, ""),
        ]
    )
    out = make(
        pd.DataFrame(), tmp_path / "r.png", phase_df=two, arms=["a", "b"], formats=("png",), dpi=40
    )
    assert out is not None and out.is_file()


def test_leading_metrics_come_first_then_alphabetical():
    table = _table(
        [
            (a, m, v, "higher")
            for a, v in (("a", 2.0), ("b", 1.0))
            for m in ("zeta", "ssim", "alpha", "psnr")
        ]
    )
    assert list(rank_table(table, "val").columns) == ["psnr", "ssim", "alpha", "zeta"]
