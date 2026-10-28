"""Tests for the EPG (Extended Phase Graph) FSE simulators.

Targets ``spectramr.infrastructure.physics.epg``. EPG is the differentiable
forward operator for fast-spin-echo simulation. The full simulator is
load-bearing for cycle-Bloch / Bloch-consistent SSDU paradigms.

Categories:

- Unit: zero PD / zero flip → zero signal.
- Property: long T2 preserves more signal than short T2 (relaxation order).
- Sanity-shape: parametrised matrix.
- Gradient: backward through both simulators reaches T1, T2.
"""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.infrastructure.physics.epg import (
    simulate_differentiable_epg_fse,
    simulate_full_epg_fse,
)

# ---------------------------------------------------------------------------
# simulate_differentiable_epg_fse
# ---------------------------------------------------------------------------


def test_a_per_row_echo_spacing_equals_each_rows_scalar_spacing() -> None:
    """The protocol render and the surrogate fit pass one spacing per row as a tensor."""
    rho = torch.full((2, 1, 2, 2), 0.8)
    t1, t2 = torch.full_like(rho, 900.0), torch.full_like(rho, 70.0)
    fa = {"excitation_fa": torch.full_like(rho, 90.0), "refocusing_fa": torch.full_like(rho, 180.0)}
    spacing = torch.tensor([2.3, 16.0]).view(2, 1, 1, 1)
    both = simulate_differentiable_epg_fse(rho, t1, t2, **fa, te_spacing=spacing)
    for row, s in enumerate((2.3, 16.0)):
        one = simulate_differentiable_epg_fse(
            rho[row : row + 1],
            t1[row : row + 1],
            t2[row : row + 1],
            excitation_fa=fa["excitation_fa"][row : row + 1],
            refocusing_fa=fa["refocusing_fa"][row : row + 1],
            te_spacing=s,
        )
        assert torch.allclose(both[row : row + 1], one)


def test_diff_epg_zero_pd_yields_zero_signal() -> None:
    """PD = 0 → output is zero."""
    rho = torch.zeros(1, 1, 4, 4)
    t1 = torch.full((1, 1, 4, 4), 1000.0)
    t2 = torch.full((1, 1, 4, 4), 80.0)
    out = simulate_differentiable_epg_fse(
        rho, t1, t2, excitation_fa=torch.tensor(90.0), refocusing_fa=torch.tensor(180.0),
        echo_train_length=8, te_spacing=10.0,
    )
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_diff_epg_zero_excitation_fa_yields_zero_signal() -> None:
    """sin(0) = 0 → no transverse magnetisation produced."""
    rho = torch.ones(1, 1, 4, 4)
    t1 = torch.full((1, 1, 4, 4), 1000.0)
    t2 = torch.full((1, 1, 4, 4), 80.0)
    out = simulate_differentiable_epg_fse(
        rho, t1, t2, excitation_fa=torch.tensor(0.0), refocusing_fa=torch.tensor(180.0),
        echo_train_length=4, te_spacing=10.0,
    )
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_diff_epg_long_t2_preserves_more_signal_than_short_t2() -> None:
    """T2 controls signal decay across echoes; long T2 → larger signal."""
    rho = torch.ones(1, 1, 4, 4)
    t1 = torch.full((1, 1, 4, 4), 1000.0)
    t2_long = torch.full((1, 1, 4, 4), 200.0)
    t2_short = torch.full((1, 1, 4, 4), 20.0)
    args = dict(
        excitation_fa=torch.tensor(90.0),
        refocusing_fa=torch.tensor(180.0),
        echo_train_length=16,
        te_spacing=10.0,
    )
    s_long = simulate_differentiable_epg_fse(rho, t1, t2_long, **args)
    s_short = simulate_differentiable_epg_fse(rho, t1, t2_short, **args)
    assert s_long.abs().mean().item() > s_short.abs().mean().item()


def test_diff_epg_output_shape_matches_rho() -> None:
    """Output preserves PD-map shape."""
    rho = torch.ones(2, 1, 16, 16)
    t1 = torch.full((2, 1, 16, 16), 800.0)
    t2 = torch.full((2, 1, 16, 16), 60.0)
    out = simulate_differentiable_epg_fse(
        rho, t1, t2, excitation_fa=torch.tensor(90.0), refocusing_fa=torch.tensor(180.0),
        echo_train_length=4,
    )
    assert out.shape == rho.shape
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# simulate_full_epg_fse
# ---------------------------------------------------------------------------


def test_full_epg_returns_four_tuple_with_correct_shapes() -> None:
    """Full EPG returns (signal[B, ETL, H, W], R1, R2, B1)."""
    rho = torch.ones(1, 1, 8, 8)
    t1 = torch.full((1, 1, 8, 8), 1000.0)
    t2 = torch.full((1, 1, 8, 8), 80.0)
    b1 = torch.ones(1, 1, 8, 8)
    etl = 4
    signal, r1, r2, b1_out = simulate_full_epg_fse(
        rho, t1, t2, b1, echo_train_length=etl,
    )
    assert signal.shape == (1, etl, 8, 8)
    assert r1.shape == rho.shape
    assert r2.shape == rho.shape
    assert b1_out.shape == b1.shape


def test_full_epg_complex_signal_output() -> None:
    """Full EPG signal is complex (carries both magnitude and phase)."""
    rho = torch.ones(1, 1, 4, 4)
    t1 = torch.full((1, 1, 4, 4), 1000.0)
    t2 = torch.full((1, 1, 4, 4), 80.0)
    b1 = torch.ones(1, 1, 4, 4)
    signal, _, _, _ = simulate_full_epg_fse(
        rho, t1, t2, b1, echo_train_length=4,
    )
    assert torch.is_complex(signal)


def test_full_epg_zero_pd_yields_zero_signal() -> None:
    """PD = 0 → signal everywhere zero."""
    rho = torch.zeros(1, 1, 4, 4)
    t1 = torch.full((1, 1, 4, 4), 1000.0)
    t2 = torch.full((1, 1, 4, 4), 80.0)
    b1 = torch.ones(1, 1, 4, 4)
    signal, _, _, _ = simulate_full_epg_fse(
        rho, t1, t2, b1, echo_train_length=4,
    )
    assert torch.allclose(signal.abs(), torch.zeros_like(signal.abs()), atol=1e-6)


def test_full_epg_r1_is_inverse_t1() -> None:
    """R1 = 1/T1 (with eps stabiliser)."""
    rho = torch.ones(1, 1, 4, 4)
    t1 = torch.full((1, 1, 4, 4), 1000.0)
    t2 = torch.full((1, 1, 4, 4), 80.0)
    b1 = torch.ones(1, 1, 4, 4)
    _, r1, r2, _ = simulate_full_epg_fse(
        rho, t1, t2, b1, echo_train_length=2,
    )
    expected_r1 = 1.0 / 1000.0
    expected_r2 = 1.0 / 80.0
    assert torch.allclose(r1, torch.full_like(r1, expected_r1), atol=1e-6)
    assert torch.allclose(r2, torch.full_like(r2, expected_r2), atol=1e-5)


# ---------------------------------------------------------------------------
# Sanity-shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [(1, 1, 8, 8), (1, 1, 16, 16), (2, 1, 16, 16)],
    ids=lambda s: "x".join(map(str, s)),
)
def test_full_epg_shape_preservation(shape: tuple[int, int, int, int]) -> None:
    rho = torch.ones(shape)
    t1 = torch.full(shape, 800.0)
    t2 = torch.full(shape, 50.0)
    b1 = torch.ones(shape)
    etl = 4
    signal, _, _, _ = simulate_full_epg_fse(
        rho, t1, t2, b1, echo_train_length=etl,
    )
    expected_shape = (shape[0], etl, shape[2], shape[3])
    assert signal.shape == expected_shape


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_full_epg_gradient_flows_to_t1_t2() -> None:
    """Backward through full EPG reaches T1 and T2."""
    rho = torch.ones(1, 1, 4, 4)
    t1 = torch.full((1, 1, 4, 4), 1000.0, requires_grad=True)
    t2 = torch.full((1, 1, 4, 4), 80.0, requires_grad=True)
    b1 = torch.ones(1, 1, 4, 4)
    signal, _, _, _ = simulate_full_epg_fse(
        rho, t1, t2, b1, echo_train_length=4,
    )
    signal.abs().sum().backward()
    assert t1.grad is not None and torch.isfinite(t1.grad).all()
    assert t2.grad is not None and torch.isfinite(t2.grad).all()


# ---------------------------------------------------------------------------
# Regression: echoes must be non-zero and follow CPMG relaxation
# ---------------------------------------------------------------------------
# Before the 2026-07 dephasing-shift fix, ``simulate_full_epg_fse`` read the
# echo from ``F_plus[0]`` *after* the trailing shift had zeroed it, so every
# echo was identically zero. The old tests (shape/dtype/zero-input) all passed
# on that all-zeros output. These tests pin the actual values.


def test_full_epg_primary_echo_matches_cpmg_relaxation() -> None:
    """90/180 CPMG primary-pathway echoes equal ρ·exp(-n·TE/T2)."""
    rho = torch.ones(1, 1, 2, 2)
    t1 = torch.full((1, 1, 2, 2), 800.0)
    t2 = torch.full((1, 1, 2, 2), 80.0)
    b1 = torch.ones(1, 1, 2, 2)
    te = 10.0
    signal, _, _, _ = simulate_full_epg_fse(
        rho, t1, t2, b1, excitation_fa=90.0, refocusing_fa=180.0,
        echo_train_length=5, te_spacing=te,
    )
    mag = signal.abs()[0, :, 0, 0]
    assert (mag > 1e-4).all(), "echoes must be non-zero (was the all-zero bug)"
    expected = torch.tensor(
        [math.exp(-(n + 1) * te / 80.0) for n in range(5)]
    )
    assert torch.allclose(mag, expected, atol=1e-3)


def test_full_epg_shorter_t2_decays_faster() -> None:
    """Shorter T2 → smaller late-echo magnitude (relaxation ordering)."""
    rho = torch.ones(1, 1, 2, 2)
    t1 = torch.full((1, 1, 2, 2), 800.0)
    b1 = torch.ones(1, 1, 2, 2)
    kw = {"refocusing_fa": 180.0, "echo_train_length": 6, "te_spacing": 10.0}
    long_t2, _, _, _ = simulate_full_epg_fse(
        rho, t1, torch.full((1, 1, 2, 2), 120.0), b1, **kw
    )
    short_t2, _, _, _ = simulate_full_epg_fse(
        rho, t1, torch.full((1, 1, 2, 2), 40.0), b1, **kw
    )
    assert short_t2.abs()[0, -1, 0, 0] < long_t2.abs()[0, -1, 0, 0]
