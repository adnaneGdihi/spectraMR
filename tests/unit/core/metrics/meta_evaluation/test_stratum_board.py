"""A stratum board takes its circular set from the whole board (stratum_board)."""

from __future__ import annotations

import numpy as np

from spectramr.core.metrics.meta_evaluation.banded_leaderboard import banded_leaderboards
from spectramr.core.metrics.meta_evaluation.damage import DAMAGE_BANDS
from spectramr.core.metrics.meta_evaluation.stratum_board import stratum_leaderboards

LEVELS = np.array([0.03, 0.05, 0.07, 0.09, 0.12, 0.18, 0.24, 0.35, 0.45, 0.6])


def test_a_stratum_board_takes_its_circular_set_from_the_whole_board() -> None:
    # "circ" tracks damage across both axes (circular); "ok" is offset per axis.
    axis = np.repeat(["x", "y", "x", "y"], LEVELS.size)
    content = np.repeat([0, 0, 1, 1], LEVELS.size)
    dmg = np.tile(LEVELS, 4) * np.repeat([1.0, 1.0, 1.1, 1.1], LEVELS.size)
    values = {"circ": dmg.copy(), "ok": dmg + np.repeat([0.0, 5.0, 0.0, 5.0], LEVELS.size)}
    hib = {"circ": False, "ok": False}
    whole = banded_leaderboards(values, axis, content, {"rel_l2": dmg}, hib)
    assert "circ" in whole["rel_l2"].circular
    one = content == 0
    res = stratum_leaderboards(
        {m: v[one] for m, v in values.items()}, axis[one], content[one], {"rel_l2": dmg[one]},
        hib, bands=DAMAGE_BANDS, whole=whole,
    )["rel_l2"]  # fmt: skip
    assert res.circular == whole["rel_l2"].circular
    assert res.circular_measured == whole["rel_l2"].circular_measured
    assert "circ" not in res.overall and "ok" in res.overall
