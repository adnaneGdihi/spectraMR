"""``comparison/group_bars.py``: raw bars from zero, arms grouped, baseline as a dotted line."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from spectramr.infrastructure.reporting.catalog.examples_payloads import make_phase_table
from spectramr.infrastructure.reporting.comparison.group_bars import _draw, make

from .test__bars import starts_at_zero

ARMS = ["a", "b"]
GROUPS = {"a": "baseline", "b": "arms"}


def _axes(phase="val", **over):
    table = make_phase_table()
    table = table[table["phase"] == phase]
    fig = _draw(
        ["psnr", "ssim"], table, over.get("arms", ARMS), over.get("groups", GROUPS), "a", phase
    )
    return fig, [a for a in fig.axes if a.get_visible()]


def test_each_bar_is_the_arms_mean_and_the_axis_contains_zero():
    fig, axes = _axes()
    table = make_phase_table()
    val = table[(table["phase"] == "val") & (table["metric"] == "psnr")].set_index("arm")["mean"]
    assert [p.get_height() for p in axes[0].patches] == [val["a"], val["b"]]
    assert all(starts_at_zero(a) for a in axes)
    plt.close(fig)


def test_the_baseline_value_is_a_dotted_line_across_the_panel():
    fig, axes = _axes()
    mean = (
        make_phase_table()
        .query("phase == 'val' and metric == 'psnr' and arm == 'a'")["mean"]
        .iloc[0]
    )
    dotted = [ln for ln in axes[0].lines if ln.get_linestyle() == ":"]
    assert len(dotted) == 1 and list(dotted[0].get_ydata()) == [mean, mean]
    plt.close(fig)


def test_validation_has_no_whisker_but_training_and_inference_carry_one_sd():
    fig, val_axes = _axes("val")
    assert not any(type(c).__name__ == "ErrorbarContainer" for c in val_axes[0].containers)
    plt.close(fig)
    for phase in ("train", "test"):
        fig, axes = _axes(phase)
        assert any(type(c).__name__ == "ErrorbarContainer" for c in axes[0].containers)
        plt.close(fig)


def test_the_groups_are_named_above_the_panel_and_an_ungrouped_arm_joins_arms():
    fig, axes = _axes()
    assert [t.get_text() for t in axes[0].texts if t.get_text() != "a"] == ["baseline", "arms"]
    plt.close(fig)
    fig, axes = _axes(groups=None)
    assert [
        t.get_text() for t in axes[0].texts if t.get_text() != "a"
    ] == []  # one group: no header
    plt.close(fig)


def test_make_writes_the_figure_and_gives_none_without_the_phase(tmp_path):
    out = make(
        pd.DataFrame(),
        tmp_path / "g.png",
        phase_df=make_phase_table(),
        arms=ARMS,
        groups=GROUPS,
        baseline="a",
        formats=("png",),
        dpi=40,
    )
    assert out is not None and out.is_file()
    only_val = make_phase_table().query("phase == 'val'")
    assert (
        make(pd.DataFrame(), tmp_path / "t.png", phase_df=only_val, arms=ARMS, phase="test") is None
    )
    assert make(pd.DataFrame(), tmp_path / "t.png", phase_df=None, arms=ARMS) is None
