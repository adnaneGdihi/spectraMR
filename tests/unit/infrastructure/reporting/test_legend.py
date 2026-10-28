"""The shared legend and palette: capped entries, stable colours, no clutter."""

import logging

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from spectramr.infrastructure.reporting import views
from spectramr.infrastructure.reporting.legend import (
    MAX_LEGEND_ENTRIES,
    arm_palette,
    collect_handles,
    reserve_legend_space,
    shared_legend,
    sparse_ticks,
)


def test_legend_is_capped_and_collapses_the_tail():
    fig, ax = plt.subplots()
    for i in range(15):
        ax.plot([0, 1], [i, i], label=f"arm{i}")
    leg = shared_legend(fig, *collect_handles([ax]))
    texts = [t.get_text() for t in leg.get_texts()]
    assert len(texts) == MAX_LEGEND_ENTRIES
    assert texts[-1] == "+4 more"
    plt.close(fig)


def test_legend_keeps_a_short_list_whole_and_skips_an_empty_one():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="a")
    assert [t.get_text() for t in shared_legend(fig, *collect_handles([ax])).get_texts()] == ["a"]
    assert shared_legend(fig, [], []) is None
    plt.close(fig)


def test_collect_handles_drops_hidden_and_duplicate_labels():
    fig, axes = plt.subplots(1, 2)
    for ax in axes:
        ax.plot([0, 1], [0, 1], label="same")
        ax.plot([0, 1], [1, 0], label="_hidden")
    assert collect_handles(axes)[1] == ["same"]
    plt.close(fig)


def test_palette_is_stable_across_calls_and_order():
    first = arm_palette(["baseline", "ours", "other"])
    again = arm_palette(["baseline", "ours", "other"])
    assert first == again
    assert len(set(first.values())) == 3


def test_twelve_unregistered_arms_get_twelve_distinct_colours():
    arms = [f"arm_{i}" for i in range(12)]
    assert len(set(arm_palette(arms).values())) == 12


def test_long_labels_use_fewer_columns_so_the_legend_stays_narrow():
    from spectramr.infrastructure.reporting.legend import legend_columns

    assert legend_columns(["a", "b", "c"]) == 3
    assert legend_columns([f"arm_{i}" for i in range(10)]) == 6
    assert legend_columns(["supervised_bound_mc_m4raw_poisson_r16"] * 12) == 2
    assert legend_columns([]) == 1


def test_legend_wraps_into_rows_and_reserves_space_for_them():
    fig, ax = plt.subplots()
    for i in range(10):
        ax.plot([0, 1], [i, i], label=f"arm{i}")
    shared_legend(fig, *collect_handles([ax]))
    height = fig.get_figheight()
    free = reserve_legend_space(fig)
    assert fig.get_figheight() == pytest.approx(height + 2 * 0.19 + 0.12)
    assert free == pytest.approx((2 * 0.19 + 0.12) / fig.get_figheight())
    plt.close(fig)


def test_sparse_ticks_caps_linear_axes_and_leaves_log_axes_alone():
    fig, (lin, log) = plt.subplots(1, 2)
    for ax in (lin, log):
        ax.plot(range(1000), range(1, 1001))
    log.set_yscale("log")
    sparse_ticks(lin, 4)
    sparse_ticks(log, 4)
    assert len(lin.get_yticks()) <= 7
    assert type(log.yaxis.get_major_locator()).__name__ == "LogLocator"
    plt.close(fig)


def _frame(with_selection):
    rows = [
        {
            "method": "m",
            "run_id": "r",
            "split": "val",
            "step": s,
            "metric": "val_psnr",
            "value": 1.0,
        }
        for s in (10, 20, 30)
    ]
    if with_selection:
        rows.append(
            {
                "method": "m",
                "run_id": "r",
                "split": "final",
                "step": -1,
                "metric": views.SELECTED_KEY,
                "value": 20.0,
            }
        )
    return pd.DataFrame(rows)


def test_recorded_steps_omits_a_run_that_recorded_no_selection():
    assert views.recorded_steps(_frame(False)) == {}
    assert views.recorded_steps(_frame(True)) == {"r": 20}


def test_selection_fallback_is_logged_once_per_run(caplog):
    views._WARNED.clear()
    with caplog.at_level(logging.WARNING, logger=views.logger.name):
        for _ in range(3):
            assert views.checkpoint_steps(_frame(False), "selected") == {"r": 30}
    assert len([r for r in caplog.records if "final validation pass" in r.message]) == 1
