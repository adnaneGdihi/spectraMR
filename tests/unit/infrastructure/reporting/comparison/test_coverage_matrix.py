"""``comparison/coverage_matrix.py``: measured, not computed, not logged."""

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from spectramr.infrastructure.reporting.catalog.examples_payloads import make_phase_table
from spectramr.infrastructure.reporting.comparison.coverage_matrix import coverage_grid, make

ARMS = ["a", "b"]


def test_a_cell_is_measured_not_computed_or_not_logged():
    table = make_phase_table()
    table = table[
        ~((table["arm"] == "b") & (table["metric"] == "ssim") & (table["phase"] == "val"))
    ]
    skipped = [{"phase": "val", "metric": "ssim", "variant": "b", "reason": "x"}]
    grid = coverage_grid(table, skipped, ARMS, "val", ["psnr", "ssim", "missing"])
    assert grid.tolist() == [[1, 1, 0], [1, 2, 0]]  # b/ssim skipped, never logged for 'missing'


def test_a_non_finite_mean_is_not_a_measurement():
    table = make_phase_table()
    table.loc[
        (table["arm"] == "a") & (table["metric"] == "psnr") & (table["phase"] == "val"), "mean"
    ] = float("nan")
    assert coverage_grid(table, [], ARMS, "val", ["psnr"]).tolist() == [[0], [1]]


def test_the_figure_is_written_and_a_wide_matrix_continues(tmp_path):
    out = make(
        pd.DataFrame(),
        tmp_path / "c.png",
        phase_df=make_phase_table(),
        arms=ARMS,
        formats=("png",),
        dpi=40,
    )
    assert out is not None and out.is_file()
    wide = pd.concat(
        [make_phase_table().assign(metric=f"m{i:02d}") for i in range(35)], ignore_index=True
    )
    make(pd.DataFrame(), tmp_path / "w.png", phase_df=wide, arms=ARMS, formats=("png",), dpi=40)
    assert (tmp_path / "w_2.png").is_file()  # 30 metrics per figure


def test_nothing_to_draw_gives_none(tmp_path):
    assert make(pd.DataFrame(), tmp_path / "c.png", phase_df=None, arms=ARMS) is None
    assert make(pd.DataFrame(), tmp_path / "c.png", phase_df=make_phase_table(), arms=[]) is None
