"""Tests for the known-protocol TSE / IR-TSE forward model (exp 32, gain-calibrated maps)."""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.infrastructure.physics.tse_forward import echo_index, tse_signal


def _maps(t1: float = 1000.0, t2: float = 100.0):
    shape = (1, 1, 2, 2)
    return torch.ones(shape), torch.full(shape, t1), torch.full(shape, t2)


def test_ideal_refocusing_reduces_to_the_spin_echo_equation() -> None:
    rho, t1, t2 = _maps()
    out = tse_signal(
        rho, t1, t2, family=0, tr=4000.0, te=96.0, ti=0.0, esp=12.0, etl=16, refocus_deg=180.0
    )
    expected = (1 - math.exp(-4.0)) * math.exp(-96.0 / 100.0)
    assert torch.allclose(out, torch.full_like(out, expected), rtol=0.05)


def test_inversion_recovery_term() -> None:
    rho, t1, t2 = _maps()
    out = tse_signal(
        rho, t1, t2, family=1, tr=7500.0, te=96.0, ti=1655.0, esp=12.0, etl=16, refocus_deg=180.0
    )
    expected = abs(1 - 2 * math.exp(-1.655) + math.exp(-7.5)) * math.exp(-0.96)
    assert torch.allclose(out, torch.full_like(out, expected), rtol=0.05)


def test_the_single_echo_t1_series_renders_at_te() -> None:
    """M4Raw's T1 is a spin echo with ETL 1; its echo spacing is TE by definition."""
    rho, t1, t2 = _maps()
    out = tse_signal(
        rho, t1, t2, family=0, tr=500.0, te=18.4, ti=0.0, esp=18.4, etl=1, refocus_deg=180.0
    )
    expected = (1 - math.exp(-0.5)) * math.exp(-18.4 / 100.0)
    assert torch.allclose(out, torch.full_like(out, expected), rtol=0.05)


def test_echo_index_out_of_train_raises() -> None:
    with pytest.raises(ValueError, match="echo"):
        echo_index(te_ms=300.0, esp_ms=12.0, etl=16)


def test_unknown_family_raises() -> None:
    rho, t1, t2 = _maps()
    with pytest.raises(ValueError, match="family"):
        tse_signal(
            rho, t1, t2, family=2, tr=500.0, te=20.0, ti=0.0, esp=20.0, etl=1, refocus_deg=180.0
        )


def test_reduced_refocusing_changes_the_signal() -> None:
    rho, t1, t2 = _maps()
    # Echo 1: by echo 8 the 180 and 120 deg trains converge to within 0.2 %.
    kw = {"family": 0, "tr": 4000.0, "te": 12.0, "ti": 0.0, "esp": 12.0, "etl": 16}
    ideal = tse_signal(rho, t1, t2, refocus_deg=180.0, **kw)
    low = tse_signal(rho, t1, t2, refocus_deg=120.0, **kw)
    assert not torch.allclose(ideal, low, rtol=0.01)


def test_b1_plus_scales_the_echo_train_flip_angles() -> None:
    rho, t1, t2 = _maps()
    kw = {"family": 0, "tr": 4000.0, "te": 24.0, "ti": 0.0, "esp": 12.0, "etl": 4}
    nominal = tse_signal(rho, t1, t2, refocus_deg=180.0, **kw)
    unit = tse_signal(rho, t1, t2, refocus_deg=180.0, b1_plus=torch.ones_like(rho), **kw)
    low = tse_signal(rho, t1, t2, refocus_deg=180.0, b1_plus=torch.full_like(rho, 0.7), **kw)
    assert torch.allclose(nominal, unit)
    assert (low < nominal).all()
