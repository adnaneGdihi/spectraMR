"""Figure A.2 renders with and without an input column.

A run with no recorded report cases falls back to its saved PNG pairs, which carry a target
and a prediction but no input (the loader refuses to invent one). The figure indexed
``case["input"]`` unconditionally and raised ``KeyError: 'input'`` on the 2026-10-10 sabine
smoke of ``experiment_32a_vae_hf_multicontrast``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spectramr.infrastructure.reporting.cases.loader import _apply_plotter_aliases
from spectramr.infrastructure.reporting.plotters.mri_specific import synthesis_c2c


def _case(*, with_input: bool) -> dict:
    rng = np.random.default_rng(0)
    arrays = {"target": rng.random((16, 16)), "prediction": rng.random((16, 16))}
    if with_input:
        arrays["input"] = rng.random((16, 16))
    return {**_apply_plotter_aliases(arrays), "case_id": "epoch001_step000010_000"}


def _spy_on_columns(monkeypatch, cases: list[dict]) -> tuple[dict, list[dict]]:
    seen = {}
    real_subplots = plt.subplots

    def spy(n_rows, n_cols, **kwargs):
        seen["n_cols"] = n_cols
        return real_subplots(n_rows, n_cols, **kwargs)

    monkeypatch.setattr(synthesis_c2c.plt, "subplots", spy)
    return seen, cases


def test_png_fallback_cases_without_input_render(tmp_path, monkeypatch) -> None:
    seen, cases = _spy_on_columns(monkeypatch, [_case(with_input=False)])
    assert "input" not in cases[0]
    out = synthesis_c2c.make(
        pd.DataFrame(), tmp_path / "c2c", cases=cases, formats=("png",), dpi=50
    )
    assert out is not None
    assert seen["n_cols"] == 5


def test_cases_with_input_keep_the_input_column(tmp_path, monkeypatch) -> None:
    seen, cases = _spy_on_columns(monkeypatch, [_case(with_input=True)])
    out = synthesis_c2c.make(
        pd.DataFrame(), tmp_path / "c2c", cases=cases, formats=("png",), dpi=50
    )
    assert out is not None
    assert seen["n_cols"] == 6
