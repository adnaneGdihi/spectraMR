"""Unit tests for the exp_11 spectral-transfer CLI driver (pure helpers, CPU-only).

The verdict function is a two-threshold classifier, so it is tested on both sides
of each threshold rather than on one comfortable example (non-negotiable 15).
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


def _load(relpath: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ST = _load("scripts/experiments/exp11_spectral_transfer_probe.py", "exp11_spectral_transfer")


def _record(label: str, outer: float):
    from spectramr.infrastructure.validation.spectral_transfer_probe import RadialTransfer

    return RadialTransfer(
        bin_edges=(0.0, 0.5, 1.0),
        bin_centers=(0.25, 0.75),
        counts=(4, 8),
        ratios=(1.0, outer),
        dc_gain=1.0,
        outer_band_retention=outer,
        outer_band_floor=outer,
        shape=(1, 4, 8, 8),
        repeats=1,
        seed=0,
        timestep=0,
        device="cpu",
        label=label,
    )


def test_discover_cohort_is_sorted(tmp_path):
    for name in ("b.yaml", "a.yaml", "c.txt"):
        (tmp_path / name).write_text("")
    assert [p.name for p in ST.discover_cohort(tmp_path)] == ["a.yaml", "b.yaml"]


@pytest.mark.parametrize(
    ("retention", "expected"),
    [
        (0.00, "LOW-PASS(architectural)"),
        (0.49, "LOW-PASS(architectural)"),
        (0.50, "attenuating"),  # the threshold itself is NOT a low-pass
        (0.79, "attenuating"),
        (0.80, "flat"),
        (1.00, "flat"),
        (3.00, "flat"),  # a high-pass is not a low-pass
    ],
)
def test_verdict_sits_on_both_sides_of_each_threshold(retention, expected):
    assert ST.verdict(retention) == expected


def test_verdict_reports_an_unmeasured_arm_rather_than_calling_it_flat():
    assert ST.verdict(None) == "UNMEASURED"
    assert ST.verdict(math.nan) == "UNMEASURED"
    assert ST.verdict(math.inf) == "UNMEASURED"


def test_ranking_puts_the_most_low_passing_arm_first():
    ranked = ST.rank_reports([_record("flat", 0.95), _record("rolled", 0.12)])
    assert [r.label for r in ranked] == ["rolled", "flat"]


def test_summary_records_the_skipped_arms_so_a_partial_run_cannot_read_as_complete():
    summary = ST.build_summary(
        [_record("a", 0.2)],
        provenance={"n_bins": 32, "repeats": 4},
        skipped=[{"arm": "b", "error": "ValueError: boom"}],
    )
    assert summary["provenance"]["repeats"] == 4
    assert summary["ranking"][0]["verdict"] == "LOW-PASS(architectural)"
    assert summary["skipped"] == [{"arm": "b", "error": "ValueError: boom"}]
    assert "normalized" in summary["arms"]["a"]


def test_summary_is_strict_json_even_with_a_non_finite_reading():
    payload = ST._json_safe(ST.build_summary([_record("a", math.nan)]))
    assert json.loads(json.dumps(payload, allow_nan=False))["ranking"][0]["verdict"] == (
        "UNMEASURED"
    )


def test_synthetic_self_check_passes_on_its_own_toy_models():
    args = ST._parse_args(["--synthetic", "--n-bins", "8", "--repeats", "1"])
    assert ST._run_synthetic(args) == 0


def test_synthetic_refuses_the_flags_it_cannot_honour():
    args = ST._parse_args(["--synthetic", "--device", "cuda"])
    with pytest.raises(ValueError, match="cannot honour"):
        ST._run_synthetic(args)
