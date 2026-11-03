"""Regression test for the 2026-06 audit fix to ``ScannerTimeReparam``.

The model held a single global parameter vector and so was architecturally
incapable of the per-scanner conditioning the cross-scanner-harmonisation
strategy advertised. It now holds a ``[num_scanners, T-1]`` table indexed by
``scanner_id`` (backward-compatible: default num_scanners=1 + no scanner_id
broadcasts the single row, as before).
"""
from __future__ import annotations

import inspect

import pytest
import torch

from spectramr.models.generators.mrf_models import ScannerTimeReparam


def test_per_scanner_tau_differs_by_scanner_id() -> None:
    m = ScannerTimeReparam(fingerprint_length=9, num_scanners=3)
    with torch.no_grad():
        # Non-constant per-scanner patterns: the Beltrami constraint normalises
        # a *constant* raw to the same identity ramp regardless of its value, so
        # the per-scanner distinction shows only for varying interval patterns.
        m.raw[0].copy_(torch.zeros(8))
        m.raw[1].copy_(torch.linspace(-1.0, 1.0, 8))
        m.raw[2].copy_(torch.linspace(1.0, -1.0, 8))
    tau = m(batch_size=2, scanner_id=torch.tensor([0, 2]))
    assert tau.shape[0] == 2
    # Distinct scanners produce distinct reparameterisations.
    assert not torch.allclose(tau[0], tau[1])


def test_backward_compatible_default_single_scanner() -> None:
    m = ScannerTimeReparam(fingerprint_length=9)  # num_scanners=1
    tau = m(batch_size=4, scanner_id=None)
    assert tau.shape[0] == 4
    # The single row is broadcast → all batch rows identical (legacy behaviour).
    assert torch.allclose(tau[0], tau[-1])


def test_rejects_non_positive_num_scanners() -> None:
    with pytest.raises(ValueError, match="num_scanners"):
        ScannerTimeReparam(num_scanners=0)


def test_strategy_threads_scanner_id() -> None:
    from spectramr.infrastructure.training.strategies.mrf_acquisition_strategies import (
        CrossScannerMRFHarmonisationStrategy,
    )

    src = inspect.getsource(
        CrossScannerMRFHarmonisationStrategy._compute_losses_impl
    )
    assert "scanner_id=scanner_id" in src


def test_out_of_range_scanner_id_raises_instead_of_collapsing() -> None:
    """An unset num_scanners must not fold every vendor onto row 0."""
    m = ScannerTimeReparam(fingerprint_length=9)  # num_scanners=1
    with pytest.raises(IndexError):
        m(batch_size=2, scanner_id=torch.tensor([0, 1]))
