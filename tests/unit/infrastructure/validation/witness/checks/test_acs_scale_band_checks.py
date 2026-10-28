"""``normalization.acs_scale_band_acquired``: green on a band inside the top
rung's guaranteed core, red on one wider than it."""

from __future__ import annotations

import pytest

from spectramr.infrastructure.validation.witness.checks.acs_scale_band_checks import (
    acs_scale_band_acquired,
)
from spectramr.infrastructure.validation.witness.registry import Severity, get_witness_registry
from spectramr.infrastructure.validation.witness.subject import WitnessSubject


def _doc(*, domain: str = "acs", fraction: float = 0.02, model: str = "kspace_cold_diffusion"):
    """The reverse cohort's mask family on a 4-rung ladder at 256x256."""
    return {
        "model": {"model_type": model},
        "data": {
            "sampling": {"patch_size": [256, 256, 1]},
            "processing": {"kspace_scale_domain": domain, "log_scaling_center_fraction": fraction},
            "acquisition": {"phase_encode_axis": -2, "acquired_pe_lines": "measured"},
        },
        "undersampling": {
            "acceleration_type": "density_nested",
            "base_acceleration": 1.0,
            "max_acceleration": 32.0,
            "center_fraction": 0.08,
            "min_center_fraction": 0.02,
            "acceleration_range": [1.0, 2.0, 8.0, 32.0],
            "schedule_type": "step",
            "schedule_steps": 4,
            "mask_seed": 42,
            "enforce_nested": True,
            "mask_direction": "phase",
        },
        "training": {"diffusion": {"timesteps": 4}},
    }


def _check(doc):
    return acs_scale_band_acquired(WitnessSubject.for_ci(None, doc))


def test_registered_at_warning_so_strict_audit_fails_on_it():
    witness = get_witness_registry().get("normalization.acs_scale_band_acquired")
    assert witness is not None
    assert witness.severity is Severity.WARNING


def _column_doc(fraction: float):
    """A 48x80 arm whose phase-encode axis is dim -1, so the band is columns (#2160)."""
    doc = _doc(fraction=fraction)
    doc["data"]["sampling"]["patch_size"] = [48, 80, 1]
    doc["data"]["acquisition"]["phase_encode_axis"] = -1
    return doc


def test_a_column_band_inside_the_core_passes_on_a_non_square_plane():
    verdict = _check(_column_doc(0.015))
    assert verdict.passed, verdict.message
    assert "columns 40..40 of 80" in verdict.message


def test_a_band_rounded_up_past_the_odd_core_fails():
    # 80 x 0.02 = 1.6 rounds to a 2-column band, 39..40; the ladder's symmetric
    # core |i - 40| <= 0.8 is column 40 alone, so the scale would move with the rung.
    verdict = _check(_column_doc(0.02))
    assert not verdict.passed
    assert "columns 39..40 of 80" in verdict.message and "column 39" in verdict.message


def test_a_column_band_wider_than_the_core_fails():
    verdict = _check(_column_doc(0.25))
    assert not verdict.passed and "columns" in verdict.message


def test_an_acs_arm_without_an_axis_fails():
    doc = _doc()
    del doc["data"]["acquisition"]
    doc["undersampling"]["mask_direction"] = "x"
    verdict = _check(doc)
    assert not verdict.passed and "phase_encode_axis" in verdict.message


def test_a_band_inside_the_guaranteed_core_passes():
    verdict = _check(_doc(fraction=0.02))
    assert verdict.passed, verdict.message
    assert "rows 126..130" in verdict.message


@pytest.mark.parametrize("fraction", [0.03, 0.25])
def test_a_band_wider_than_the_core_fails(fraction):
    verdict = _check(_doc(fraction=fraction))
    assert not verdict.passed
    assert verdict.severity is Severity.WARNING
    assert "misses" in verdict.message


@pytest.mark.parametrize(
    ("doc", "phrase"),
    [(_doc(domain="image"), "not 'acs'"), (_doc(model="unet"), "no k-space ladder")],
)
def test_it_steps_aside_where_it_has_nothing_to_check(doc, phrase):
    verdict = _check(doc)
    assert verdict.passed
    assert verdict.severity is Severity.INFO
    assert phrase in verdict.message
