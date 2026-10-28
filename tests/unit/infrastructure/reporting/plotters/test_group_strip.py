"""Tests for the QC group IQM distribution plotter.

Targets ``spectramr.infrastructure.reporting.plotters.qc.group_strip`` via the
plotter registry. Verifies it renders from a wide per-case frame, falls back to
a long predictions frame, soft-skips on empty input, and never touches the
global RNG.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from spectramr.infrastructure.reporting import plotters


def _wide(n=20):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "case_id": [f"val_step{i}" for i in range(n)],
            "split": ["val"] * n,
            "step": [100] * n,
            "batch_index": list(range(n)),
            "psnr": rng.normal(30, 2, n),
            "ssim": rng.uniform(0.7, 0.95, n),
        }
    )


def test_registered():
    assert "qc_group_strip" in plotters.list_available()


def test_renders_from_wide_per_case_frame(tmp_path):
    out = tmp_path / "grp.pdf"
    res = plotters.get("qc_group_strip")(
        pd.DataFrame(), out, per_call_df=_wide(), formats=("pdf",)
    )
    assert res is not None and res.exists()


def test_renders_from_long_predictions_frame(tmp_path):
    long = pd.DataFrame(
        {
            "method": ["model"] * 8,
            "metric": ["psnr"] * 4 + ["ssim"] * 4,
            "value": [30, 31, 29, 40, 0.8, 0.82, 0.79, 0.5],
            "subject_id": [f"c{i}" for i in range(4)] * 2,
            "split": ["test"] * 8,
        }
    )
    res = plotters.get("qc_group_strip")(
        pd.DataFrame(), tmp_path / "g.pdf", predictions_df=long, formats=("pdf",)
    )
    assert res is not None and res.exists()


def test_soft_skip_when_no_metric_data(tmp_path):
    res = plotters.get("qc_group_strip")(
        pd.DataFrame(), tmp_path / "g.pdf", formats=("pdf",)
    )
    assert res is None


def test_does_not_touch_global_rng(tmp_path):
    np.random.seed(123)
    before = np.random.get_state()[1][0]
    plotters.get("qc_group_strip")(
        pd.DataFrame(), tmp_path / "g.pdf", per_call_df=_wide(), formats=("pdf",)
    )
    after = np.random.get_state()[1][0]
    assert before == after


# ---------------------------------------------------------------------------
# #503: a distribution that was never measured.
#
# On exp_vf_01 the per-case sink received the BATCH AGGREGATE — one row per
# validation call, not per sample — so eight "cases" all carried the single
# value 48.71364215. This plotter rendered them as a box-and-whisker while its
# siblings on the same report honestly reported "skipped (no data)".
#
# The subtle half is `max(q3 - q1, 1e-9)`: with n < 4 the code set
# q1 = med = q3, and that clamp turned an invisible zero-width box into a thin
# SLIVER, which reads as an extremely tight distribution — the most confident
# claim on the chart, from the least data.
# ---------------------------------------------------------------------------


def _flat(n=8, value=48.71364215):
    """The exp_vf_01 shape: one scalar repeated under n distinct case_ids."""
    return pd.DataFrame(
        {
            "case_id": [f"val_step{i}" for i in range(n)],
            "split": ["val"] * n,
            "step": [100] * n,
            "batch_index": list(range(n)),
            "psnr": [value] * n,
        }
    )


class TestDegenerateSeriesAreNotDrawn:
    def test_a_repeated_scalar_skips_the_figure(self, tmp_path):
        """The headline case: eight identical values must produce NO figure."""
        res = plotters.get("qc_group_strip")(
            pd.DataFrame(), tmp_path / "g.pdf", per_call_df=_flat(), formats=("pdf",)
        )
        assert res is None

    def test_a_single_observation_is_not_a_distribution(self, tmp_path):
        res = plotters.get("qc_group_strip")(
            pd.DataFrame(), tmp_path / "g.pdf", per_call_df=_flat(n=1), formats=("pdf",)
        )
        assert res is None

    def test_the_skip_is_logged_not_silent(self, tmp_path, caplog):
        """A missing panel must be explainable, or it reads as a broken plotter."""
        import logging

        with caplog.at_level(logging.WARNING):
            plotters.get("qc_group_strip")(
                pd.DataFrame(),
                tmp_path / "g.pdf",
                per_call_df=_flat(),
                formats=("pdf",),
            )
        assert any(
            "psnr" in r.getMessage() and "no spread" in r.getMessage()
            for r in caplog.records
        )

    def test_a_flat_metric_is_dropped_but_a_varying_sibling_still_renders(
        self, tmp_path
    ):
        """Per-metric, not all-or-nothing: one dead column must not kill the panel."""
        frame = _flat()
        frame["ssim"] = np.linspace(0.70, 0.95, len(frame))
        res = plotters.get("qc_group_strip")(
            pd.DataFrame(), tmp_path / "g.pdf", per_call_df=frame, formats=("pdf",)
        )
        assert res is not None and res.exists()

    def test_genuine_spread_still_renders(self, tmp_path):
        """Anti-vacuity: the guard must not simply disable the plotter."""
        res = plotters.get("qc_group_strip")(
            pd.DataFrame(), tmp_path / "g.pdf", per_call_df=_wide(), formats=("pdf",)
        )
        assert res is not None and res.exists()


class TestNoFabricatedQuartiles:
    """Below n=4 there are no quartiles, so no box may be drawn."""

    @staticmethod
    def _n_boxes(vals):
        import matplotlib.pyplot as plt

        from spectramr.infrastructure.reporting.plotters.qc.group_strip import _draw_hbox

        fig, ax = plt.subplots()
        try:
            _draw_hbox(ax, np.asarray(vals, dtype=float))
            return sum(isinstance(p, plt.Rectangle) for p in ax.patches)
        finally:
            plt.close(fig)

    def test_three_points_draw_no_box(self):
        assert self._n_boxes([1.0, 2.0, 9.0]) == 0

    def test_four_points_draw_the_box(self):
        assert self._n_boxes([1.0, 2.0, 3.0, 9.0]) == 1

    def test_the_median_is_still_shown_below_the_threshold(self):
        """Dropping the box must not drop what the data DOES say."""
        import matplotlib.pyplot as plt

        from spectramr.infrastructure.reporting.plotters.qc.group_strip import _draw_hbox

        fig, ax = plt.subplots()
        try:
            _draw_hbox(ax, np.array([1.0, 2.0, 9.0]))
            xs = [tuple(line.get_xdata()) for line in ax.lines]
        finally:
            plt.close(fig)
        assert (2.0, 2.0) in xs, "median tick missing"
        assert (1.0, 9.0) in xs, "min-max span missing"


class TestHasSpread:
    @pytest.mark.parametrize(
        ("vals", "expected"),
        [
            ([], False),
            ([1.0], False),
            ([2.5, 2.5, 2.5], False),
            ([1.0, 1.0000001], True),
            ([0.0, 1.0], True),
        ],
    )
    def test_predicate(self, vals, expected):
        from spectramr.infrastructure.reporting.plotters.qc.group_strip import _has_spread

        assert _has_spread(np.asarray(vals, dtype=float)) is expected


# ---------------------------------------------------------------------------
# per_call_metrics.csv carries every validation PASS. A strip pooled across
# passes draws the training trajectory as if it were spread across cases, and
# pooling R2/R8/R32 draws three distributions as one.
# ---------------------------------------------------------------------------


def _passes():
    """Two passes: an early one far from the late one, 6 calls each."""
    rows = []
    for step, centre in ((100, 10.0), (200, 30.0)):
        for b in range(6):
            rows.append(
                {
                    "case_id": f"val_step{step}",
                    "split": "val",
                    "step": step,
                    "batch_index": b,
                    "psnr": centre + 0.1 * b,
                }
            )
    return pd.DataFrame(rows)


class TestOnePassOnly:
    def test_reads_the_last_pass_not_the_trajectory(self):
        from spectramr.infrastructure.reporting.plotters.qc import group_strip

        vals = group_strip._collect(pd.DataFrame(), _passes(), None)["psnr"][0]
        assert vals.size == 6 and vals.min() >= 30.0

    def test_reads_the_early_stopping_pass_when_the_run_recorded_one(self):
        from spectramr.infrastructure.reporting.plotters.qc import group_strip

        df = pd.DataFrame(
            [
                {"run_id": "r", "split": "val", "step": 100, "metric": "val_psnr", "value": 11.0},
                {"run_id": "r", "split": "val", "step": 200, "metric": "val_psnr", "value": 9.0},
                {
                    "run_id": "r",
                    "split": "final",
                    "step": -1,
                    "metric": "early_stopping_best_iteration",
                    "value": 100.0,
                },
            ]
        )
        vals = group_strip._collect(df, _passes(), None)["psnr"][0]
        assert vals.max() < 11.0

    def test_the_aggregator_trajectory_alone_is_no_data(self, tmp_path):
        df = pd.DataFrame(
            [
                {"split": "val", "step": s, "metric": "val_psnr", "value": float(s)}
                for s in range(10)
            ]
        )
        assert plotters.get("qc_group_strip")(df, tmp_path / "g.pdf", formats=("pdf",)) is None


def test_accelerations_get_their_own_rows(tmp_path):
    from spectramr.infrastructure.reporting.plotters.qc import group_strip

    frame = pd.DataFrame(
        [
            {
                "case_id": "c",
                "split": "val",
                "step": 1,
                "batch_index": b,
                "acceleration_level": r,
                "heldout": False,
                "psnr": 40.0 - r + 0.1 * b,
            }
            for r in (2.0, 8.0)
            for b in range(5)
        ]
    )
    series = group_strip._collect(pd.DataFrame(), frame, None)
    assert set(series["psnr"][2]) == {2.0, 8.0}
    assert "heldout" not in series
    res = plotters.get("qc_group_strip")(
        pd.DataFrame(), tmp_path / "g.pdf", per_call_df=frame, formats=("pdf",)
    )
    assert res is not None and res.exists()
