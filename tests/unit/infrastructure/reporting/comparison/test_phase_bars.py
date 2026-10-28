"""``comparison/phase_bars.py``: raw bars by phase, inference hatched, whiskers only where there is spread."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from spectramr.infrastructure.reporting.catalog.examples_payloads import make_phase_table
from spectramr.infrastructure.reporting.comparison._bars import HATCH
from spectramr.infrastructure.reporting.comparison.phase_bars import _draw, make

from .test__bars import starts_at_zero

ARMS = ["a", "b"]


def _fig(table=None):
    return _draw(["psnr"], make_phase_table() if table is None else table, ARMS)


def test_one_bar_per_arm_and_phase_all_from_zero():
    fig = _fig()
    ax = next(a for a in fig.axes if a.get_visible())
    assert len(ax.patches) == 6  # 2 arms x 3 phases
    assert starts_at_zero(ax)
    plt.close(fig)


def test_only_the_inference_bars_are_hatched_and_each_arm_keeps_its_colour():
    fig = _fig()
    ax = next(a for a in fig.axes if a.get_visible())
    hatched = [p for p in ax.patches if p.get_hatch() == HATCH]
    assert len(hatched) == 2
    by_arm = {}
    for i, p in enumerate(ax.patches):
        by_arm.setdefault(i % 2, set()).add(p.get_facecolor())
    assert all(len(colours) == 1 for colours in by_arm.values())  # same arm, same colour
    plt.close(fig)


def test_the_single_validation_pass_has_no_whisker_the_other_phases_do():
    fig = _fig()
    ax = next(a for a in fig.axes if a.get_visible())
    errorbars = [c for c in ax.containers if type(c).__name__ == "ErrorbarContainer"]
    assert len(errorbars) == 4  # train and test, two arms each; validation has none
    plt.close(fig)


def test_phases_are_named_once_on_the_axis_not_also_as_headers():
    fig = _fig()
    ax = next(a for a in fig.axes if a.get_visible())
    names = [t.get_text().split("\n")[0] for t in ax.get_xticklabels()]
    assert names == ["training", "validation", "inference"]
    assert [t.get_text() for t in ax.texts if t.get_text() != "a"] == []
    plt.close(fig)


def test_make_writes_the_figure_and_handles_nothing_to_draw(tmp_path):
    out = make(
        pd.DataFrame(),
        tmp_path / "p.png",
        phase_df=make_phase_table(),
        arms=ARMS,
        formats=("png",),
        dpi=40,
    )
    assert out is not None and out.is_file()
    empty = make_phase_table().assign(mean=float("nan"))
    assert make(pd.DataFrame(), tmp_path / "e.png", phase_df=empty, arms=ARMS) is None
