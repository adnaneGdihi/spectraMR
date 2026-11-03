"""Tests for the super-Nyquist fidelity metric.

The band arithmetic itself is covered in
``tests/unit/infrastructure/physics/test_band_partition.py``. What is asserted
here is the metric CONTRACT: registry visibility, direction, the passband
requirement, mask caching, and that the reported spectrum keeps the per-band
shape rather than collapsing it to one number.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.core.metrics.registry import get_metric, list_available  # noqa: E402
from spectramr.core.metrics.super_nyquist_fidelity import (  # noqa: E402
    SuperNyquistFidelity,
)


def _pair(h: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    target = torch.randn(2, 1, h, h)
    return target.clone(), target


# ── registry contract ────────────────────────────────────────────────────────


def test_metric_is_registered_and_reachable_by_alias() -> None:
    assert "super_nyquist_fidelity" in list_available()
    assert isinstance(get_metric("snf", sr_scale=2), SuperNyquistFidelity)


def test_metric_declares_higher_is_better() -> None:
    """A hallucination certificate that is silently graded 'lower is better'
    would invert every leaderboard it appears on."""
    assert get_metric("super_nyquist_fidelity", sr_scale=2).higher_is_better is True


# ── the passband must be declared, never assumed ─────────────────────────────


def test_construction_without_a_passband_raises() -> None:
    """With no Nyquist boundary, 'super-Nyquist' has no referent."""
    with pytest.raises(ValueError, match="needs a passband"):
        SuperNyquistFidelity()


def test_millimetre_passband_is_accepted() -> None:
    snf = SuperNyquistFidelity(voxel_mm=(0.49, 0.49), effective_voxel_mm=(1.6, 1.6), rho_max=2.0)
    assert snf.super_nyquist_bands == (2, 3)
    pred, target = _pair(128)
    assert -1.0 <= snf(pred, target) <= 1.0


def test_edges_and_super_band_indices_are_exposed() -> None:
    snf = SuperNyquistFidelity(sr_scale=2, n_sub_bands=3, n_super_bands=1, rho_max=1.5)
    assert snf.edges == pytest.approx((0.0, 1 / 3, 2 / 3, 1.0, 1.5))
    assert snf.super_nyquist_bands == (3,)


# ── values ───────────────────────────────────────────────────────────────────


def test_identical_inputs_score_one() -> None:
    snf = SuperNyquistFidelity(sr_scale=2)
    pred, target = _pair()
    assert snf(pred, target) == pytest.approx(1.0, abs=1e-5)


def test_independent_inputs_score_near_zero() -> None:
    snf = SuperNyquistFidelity(sr_scale=2)
    _, target = _pair()
    assert abs(snf(torch.randn_like(target), target)) < 0.25


def test_complex_inputs_are_reduced_to_magnitude() -> None:
    snf = SuperNyquistFidelity(sr_scale=2)
    _, target = _pair()
    cplx = torch.complex(target, torch.zeros_like(target))
    assert snf(cplx, cplx) == pytest.approx(1.0, abs=1e-5)


def test_call_does_not_leak_gradient_but_transfer_does() -> None:
    """``__call__`` reports, ``transfer`` trains. Conflating them would either
    build a graph through every validation step or hand the probe a detached
    tensor and silently make the loss term inert."""
    snf = SuperNyquistFidelity(sr_scale=2)
    _, target = _pair(32)
    pred = torch.randn_like(target).requires_grad_(True)
    assert isinstance(snf(pred, target), float)
    out = snf.transfer(pred, target)
    assert out.requires_grad
    out.mean().backward()
    assert pred.grad is not None and float(pred.grad.norm()) > 0.0


# ── reporting ────────────────────────────────────────────────────────────────


def test_spectrum_reports_every_band_plus_the_two_summaries() -> None:
    """The rolloff SHAPE is the finding; a single mean hides it."""
    snf = SuperNyquistFidelity(sr_scale=2, n_sub_bands=2, n_super_bands=2)
    pred, target = _pair()
    spec = snf.spectrum(pred, target)
    assert sum(k.startswith("snf_band_") for k in spec) == 4
    assert spec["snf_sub_nyquist"] == pytest.approx(1.0, abs=1e-5)
    assert spec["snf_super_nyquist"] == pytest.approx(1.0, abs=1e-5)


def test_spectrum_keys_carry_the_band_lower_edge() -> None:
    """A reader must be able to tell which band a number belongs to without
    reconstructing the partition from the config."""
    snf = SuperNyquistFidelity(sr_scale=2, n_sub_bands=2, n_super_bands=2)
    pred, target = _pair()
    keys = [k for k in snf.spectrum(pred, target) if k.startswith("snf_band_")]
    assert keys == [
        "snf_band_0_lo0.00",
        "snf_band_1_lo0.50",
        "snf_band_2_lo1.00",
        "snf_band_3_lo1.50",
    ]


def test_spectrum_prefix_is_configurable() -> None:
    snf = SuperNyquistFidelity(sr_scale=2)
    pred, target = _pair()
    assert all(k.startswith("probe_") for k in snf.spectrum(pred, target, prefix="probe"))


# ── caching ──────────────────────────────────────────────────────────────────


def test_masks_are_cached_per_grid_and_rebuilt_for_a_new_one() -> None:
    snf = SuperNyquistFidelity(sr_scale=2)
    a = torch.randn(1, 1, 64, 64)
    b = torch.randn(1, 1, 32, 32)
    assert snf.masks(a) is snf.masks(a)
    assert snf.masks(b) is not snf.masks(a)
    assert snf.masks(b).shape[1:] == (32, 32)


def test_underpopulated_band_raises_rather_than_reporting_noise() -> None:
    """A band holding a handful of bins still yields a number, and that number
    reads as a transfer measurement it is not."""
    snf = SuperNyquistFidelity(sr_scale=1, rho_max=2.0)
    with pytest.raises(ValueError, match="frequency bins"):
        snf(*_pair(32))


def test_unknown_geometry_raises_at_construction() -> None:
    with pytest.raises(ValueError, match="geometry"):
        SuperNyquistFidelity(sr_scale=2, geometry="spiral")  # type: ignore[arg-type]
