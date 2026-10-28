"""The pass value of a ratio metric is averaged on its reciprocal scale.

``val_nex_equivalent`` is a per-slice ratio, floor / corrected MSE. A low-signal
slice whose corrected MSE is near zero reads a huge N_eq and dominated the
arithmetic pass mean. PAPER 5.3 and the per-slice reader (``per_slice.RECIPROCAL``)
average ``1/N_eq``; the pass value now does too.
"""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.pipelines import train as train_mod


def _pass(rows: list[dict]) -> dict[str, float]:
    accum: dict = {}
    finite: dict = {}
    seen: dict = {}
    for row in rows:
        train_mod._accumulate_finite_metrics(accum, finite, seen, row, 1)
    return {k: train_mod._pass_mean(k, accum[k], finite[k]) for k in accum}


def test_nex_equivalent_is_the_reciprocal_mean() -> None:
    rows = [
        {"val_nex_equivalent": 2.0, "val_psnr": 20.0},
        {"val_nex_equivalent": 1000.0, "val_psnr": 22.0},
    ]
    out = _pass(rows)
    assert out["val_nex_equivalent"] == pytest.approx(1.0 / ((0.5 + 0.001) / 2))
    assert out["val_psnr"] == pytest.approx(21.0)  # other columns unchanged


def test_tensor_values_take_the_same_route() -> None:
    rows = [{"val_nex_equivalent": torch.tensor(4.0)}, {"val_nex_equivalent": torch.tensor(4.0)}]
    assert _pass(rows)["val_nex_equivalent"] == pytest.approx(4.0)


def test_an_infinite_ratio_is_a_finite_reciprocal() -> None:
    """N_eq = inf (zero corrected MSE) is 1/N_eq = 0, a legitimate value."""
    rows = [{"val_nex_equivalent": math.inf}, {"val_nex_equivalent": 2.0}]
    assert _pass(rows)["val_nex_equivalent"] == pytest.approx(4.0)


def test_planted_arithmetic_mean_is_dominated_by_one_slice() -> None:
    values = [2.0, 1000.0]
    arithmetic = sum(values) / len(values)
    reciprocal = _pass([{"val_nex_equivalent": v} for v in values])["val_nex_equivalent"]
    assert arithmetic > 100 * reciprocal / 2
