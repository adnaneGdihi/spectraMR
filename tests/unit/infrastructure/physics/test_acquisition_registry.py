"""Tests for ``AcquisitionRegistry``.

Targets ``spectramr.infrastructure.physics.acquisition_registry``. Looks up
TR/TE/TI/B0 protocol parameters by ``(contrast, field_strength)`` and
returns a min-max-normalised ``[1, 4]`` tensor for downstream conditioning.

Categories:

- Direct keys: ``T1w_64mT``, ``T2w_3T``, ``FLAIR_3T`` etc. lookup
- Normalisation: TR/TE/TI/B0 each in ``[0, 1]`` after dividing by class-level max
- Field-strength alias: ``"3T" / "3.0T" / "High"`` all route to the same protocol
- Unknown key falls back to ``T1w_3T`` or ``T1w_64mT`` based on field strength
- ``device`` argument respected
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.acquisition_registry import AcquisitionRegistry


# ---------------------------------------------------------------------------
# Direct lookup
# ---------------------------------------------------------------------------


def test_t1w_3t_direct_lookup() -> None:
    """``(T1w, 3T)`` returns a normalised tensor of shape ``[1, 4]``."""
    vec = AcquisitionRegistry.get_physics_vector("T1w", "3T")
    assert vec.shape == (1, 4)
    # B0 = 3T / max 3T = 1.0
    assert pytest.approx(vec[0, 3].item()) == 1.0


def test_flair_3t_normalises_correctly() -> None:
    """FLAIR_3T: TR=11000 / 11000 = 1.0 (max-saturation)."""
    vec = AcquisitionRegistry.get_physics_vector("FLAIR", "3T")
    # Position 0 is TR.
    assert pytest.approx(vec[0, 0].item()) == 1.0


def test_t2w_64mt_low_field_normalisation() -> None:
    """``(T2w, 64mT)`` normalises B0 = 0.064 / 3 = 0.0213…"""
    vec = AcquisitionRegistry.get_physics_vector("T2w", "64mT")
    expected_b0 = 0.064 / AcquisitionRegistry.MAX_B0
    assert pytest.approx(vec[0, 3].item(), abs=1e-5) == expected_b0


# ---------------------------------------------------------------------------
# Normalisation bounds
# ---------------------------------------------------------------------------


def test_all_protocols_within_unit_range() -> None:
    """Every supported protocol has a normalised vector in ``[0, 1]``."""
    for contrast in ("T1w", "T2w", "FLAIR"):
        for field in ("64mT", "3T"):
            vec = AcquisitionRegistry.get_physics_vector(contrast, field)
            assert (vec >= 0).all(), f"negative entries for {contrast}_{field}"
            assert (vec <= 1.0 + 1e-6).all(), f"oversaturated for {contrast}_{field}"


# ---------------------------------------------------------------------------
# Field-strength aliases
# ---------------------------------------------------------------------------


def test_3T_aliases_dispatch_to_same_vector() -> None:
    """``3T`` and ``High`` (and ``3.0T``) route to the same protocol."""
    a = AcquisitionRegistry.get_physics_vector("T1w", "3T")
    b = AcquisitionRegistry.get_physics_vector("T1w", "High")
    c = AcquisitionRegistry.get_physics_vector("T1w", "3.0T")
    assert torch.allclose(a, b)
    assert torch.allclose(a, c)


def test_64mT_aliases_dispatch_to_same_vector() -> None:
    """``64mT`` / ``0.064T`` / ``Low`` all alias to the same low-field protocol."""
    a = AcquisitionRegistry.get_physics_vector("T1w", "64mT")
    b = AcquisitionRegistry.get_physics_vector("T1w", "0.064T")
    c = AcquisitionRegistry.get_physics_vector("T1w", "Low")
    assert torch.allclose(a, b)
    assert torch.allclose(a, c)


# ---------------------------------------------------------------------------
# Unknown key fallback
# ---------------------------------------------------------------------------


def test_unknown_contrast_high_field_falls_back() -> None:
    """Unknown contrast + ``3T`` falls back to ``T1w_3T``."""
    vec = AcquisitionRegistry.get_physics_vector("UNKNOWN", "3T")
    expected = AcquisitionRegistry.get_physics_vector("T1w", "3T")
    assert torch.allclose(vec, expected)


def test_unknown_contrast_low_field_falls_back() -> None:
    """Unknown contrast + low-field falls back to ``T1w_64mT``."""
    vec = AcquisitionRegistry.get_physics_vector("UNKNOWN", "Low")
    expected = AcquisitionRegistry.get_physics_vector("T1w", "Low")
    assert torch.allclose(vec, expected)


# ---------------------------------------------------------------------------
# Device argument
# ---------------------------------------------------------------------------


def test_device_argument_respected() -> None:
    """The returned tensor lives on the requested device."""
    vec = AcquisitionRegistry.get_physics_vector("T1w", "3T", device="cpu")
    assert vec.device.type == "cpu"


# ---------------------------------------------------------------------------
# Class-level max constants
# ---------------------------------------------------------------------------


def test_max_constants_are_documented_values() -> None:
    """Sanity-check the documented MAX values used by the normaliser."""
    assert AcquisitionRegistry.MAX_TR == 11000.0
    assert AcquisitionRegistry.MAX_TE == 200.0
    assert AcquisitionRegistry.MAX_TI == 3000.0
    assert AcquisitionRegistry.MAX_B0 == 3.0


# ---------------------------------------------------------------------------
# Strict per-row lookup
# ---------------------------------------------------------------------------


def test_physics_vectors_for_fields_match_per_row_protocol() -> None:
    """Each row gets the vector of its own field; no row borrows another's."""
    rows = AcquisitionRegistry.physics_vectors_for_fields("T1w", torch.tensor([3.0, 0.064, 3.0]))
    assert rows.shape == (3, 4)
    assert torch.allclose(rows[0], AcquisitionRegistry.get_physics_vector("T1w", "3T")[0])
    assert torch.allclose(rows[1], AcquisitionRegistry.get_physics_vector("T1w", "64mT")[0])


def test_physics_vectors_for_unregistered_field_raises() -> None:
    """0.3 T has no protocol; the strict lookup refuses instead of borrowing 64 mT."""
    with pytest.raises(ValueError, match="no registered protocol"):
        AcquisitionRegistry.physics_vectors_for_fields("T1w", torch.tensor([0.3]))


def test_physics_vectors_for_unregistered_contrast_raises() -> None:
    with pytest.raises(ValueError, match="no registered protocol for contrast 'ADC'"):
        AcquisitionRegistry.physics_vectors_for_fields("ADC", torch.tensor([3.0]))
