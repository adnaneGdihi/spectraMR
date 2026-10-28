"""Regression: the main-results table used a local higher-is-better heuristic.

2026-06-11 infrastructure audit. ``main_results.make`` had its own substring
heuristic that defaulted unknown metrics to higher-is-better, so a lower-better
metric whose name lacked a hardcoded keyword (``cjv``, ``efc``,
``bland_altman_bias``, ``ghosting_ratio``, …) had its WORST run bolded as best.
Direction now comes from ``METRIC_HIGHER_IS_BETTER`` (the same SSOT the
pipelines layer uses).
"""

from __future__ import annotations

import pandas as pd

from spectramr.infrastructure.reporting.tables import main_results


def _df():
    # cjv is lower-is-better in the SSOT but lacks a heuristic keyword.
    return pd.DataFrame(
        [
            {"method": "methodA", "metric": "cjv", "value": 0.100, "split": "test"},
            {"method": "methodB", "metric": "cjv", "value": 0.900, "split": "test"},
        ]
    )


def test_lower_better_metric_bolds_the_lower_value(tmp_path):
    out = main_results.make(_df(), tmp_path, metrics=["cjv"])
    assert out is not None
    md = (tmp_path / "tab_2_1_main_results.md").read_text()
    rows = {
        line.split("|")[1].strip(): line for line in md.splitlines() if line.startswith("| method")
    }
    # methodA (0.100, the better/lower) must be the bolded best.
    assert "**0.100**" in rows["methodA"]
    # methodB (0.900, worse) must NOT be bolded as best.
    assert "**0.900**" not in rows["methodB"]


def _two_methods(metric: str, a: float, b: float, split: str = "test") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"method": "methodA", "metric": metric, "value": a, "split": split},
            {"method": "methodB", "metric": metric, "value": b, "split": split},
        ]
    )


def _rows(tmp_path) -> dict[str, str]:
    md = (tmp_path / "tab_2_1_main_results.md").read_text()
    return {ln.split("|")[1].strip(): ln for ln in md.splitlines() if ln.startswith("| method")}


def test_a_strategy_emitted_lower_is_better_key_bolds_its_lower_value(tmp_path):
    """The swap error is declared lower-is-better outside the registry map.

    The table read only METRIC_HIGHER_IS_BETTER and then a keyword heuristic that
    defaults to higher, so it bolded the WORST swap error as best.
    """
    # Rows are keyed by base name (the split prefix is stripped), so the request is too.
    main_results.make(
        _two_methods("val_swap_content_distance", 0.2, 0.7), tmp_path,
        metrics=["swap_content_distance"],
    )  # fmt: skip
    rows = _rows(tmp_path)
    assert "**0.200**" in rows["methodA"]
    assert "**0.700**" not in rows["methodB"]


def test_a_declared_non_registry_key_takes_its_declared_direction(tmp_path):
    """val_t1_mape is declared lower-is-better; the old heuristic guessed higher."""
    main_results.make(_two_methods("val_t1_mape", 3.0, 9.0), tmp_path, metrics=["t1_mape"])
    assert "**3.000**" in _rows(tmp_path)["methodA"]


def test_the_disentanglement_readings_join_the_default_selection_when_present(tmp_path):
    df = pd.concat(
        [
            _two_methods("psnr", 30.0, 31.0),
            _two_methods("val_style_contrast_acc", 0.9, 0.6),
            _two_methods("val_content_contrast_leak", 0.05, 0.4),
            _two_methods("val_swap_content_distance", 0.2, 0.7),
            _two_methods("val_swap_style_distance", 0.3, 0.1),
            _two_methods("val_style_contrast_chance", 0.33, 0.33),  # a reference, not a result
        ],
        ignore_index=True,
    )
    main_results.make(df, tmp_path)
    header = (tmp_path / "tab_2_1_main_results.md").read_text().splitlines()[0].lower()
    for label in (
        "style contrast acc ↑",
        "content contrast leak ↓",
        "swap content distance ↓",
        "swap style distance ↓",
    ):
        assert label in header, header
    assert "chance" not in header
    rows = _rows(tmp_path)
    assert "**0.900**" in rows["methodA"]  # style accuracy: higher wins
    assert "**0.050**" in rows["methodA"]  # content leak: lower wins
    assert "**0.100**" in rows["methodB"]  # swap style error: lower wins
