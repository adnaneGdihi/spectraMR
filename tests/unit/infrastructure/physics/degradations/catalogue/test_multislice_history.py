"""spin_history: through-plane motion carries each source slice's magnetisation history."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from spectramr.infrastructure.physics.degradations import (
    AcquisitionGeometry,
    AcquisitionMode,
    SliceOrder,
    apply_degradation,
)
from spectramr.infrastructure.physics.degradations.blocks.acquisition_timing import (
    line_times,
    shot_index,
)
from spectramr.infrastructure.physics.degradations.catalogue.multislice_common import (
    hop_intervals,
    intervals_at,
)
from spectramr.infrastructure.physics.degradations.catalogue.multislice_history import (
    excess_gain,
    history_multislice,
    history_volume,
    lines_since_hop,
    mz_after_n,
    shot_times,
    steady_state,
)
from spectramr.infrastructure.physics.scanner_presets import FASTMRI_3T

from .invariants import GEOMETRIES, assert_axis_invariants

NAME = "spin_history"
M = AcquisitionMode
TR, T1, FA = 0.05, 1.0, 30.0
MS = AcquisitionGeometry(mode=M.MULTISLICE_2D, voxel_mm=(5.0, 1.0, 1.0))


A = math.radians(FA)
E1 = math.exp(-TR / T1)
M_SS = (1 - E1) / (1 - E1 * math.cos(A))  # fixed point of M -> 1 + (M cos a - 1) E1


def _mz(dt, n):
    """Hand recursion: relax from M_ss cos a for dt, then n excitation/relaxation steps."""
    m = 1 - (1 - M_SS * math.cos(A)) * (math.exp(-dt / T1) if math.isfinite(dt) else 0.0)
    for _ in range(n):
        m = 1 + (m * math.cos(A) - 1) * E1
    return m


def _gain(dt, n):
    return _mz(dt, n) / M_SS


@pytest.mark.parametrize("geom", ["ms_interleaved", "ms_sequential", "volume", "volume_centric"])
def test_invariants(geom):
    assert_axis_invariants(NAME, GEOMETRIES[geom])


def test_hop_intervals_follow_the_interleaved_excitation_order():
    # Four slices interleave 0, 2, 1, 3, i.e. ranks 0, 2, 1, 3 (fire times rank/4 TR): slice 1 fires
    # 2/4 TR after slice 0, slice 2 fires 1/4 TR before slice 1 so 3/4 TR after it (previous cycle),
    # slice 3 fires 2/4 TR after slice 2.
    dt = hop_intervals(MS, 4, 16, TR, (-1, 0))
    assert dt[0].tolist() == pytest.approx([math.inf, 0.5 * TR, 0.75 * TR, 0.5 * TR], rel=1e-5)
    assert dt[1].tolist() == pytest.approx([TR] * 4, rel=1e-5)


def test_sequential_hop_is_the_block_distance():
    seq = replace(MS, slice_order=SliceOrder.SEQUENTIAL)
    dt = hop_intervals(seq, 3, 16, TR, (-1,))
    assert dt[0, 1] == pytest.approx((16 // 2 + 1) * TR, rel=1e-5)  # centre line to block end


@pytest.mark.parametrize("n", [0, 1, 5])
def test_mz_after_n_matches_the_hand_recursion(n):
    for dt in (0.2 * TR, 0.75 * TR, 3.0, math.inf):
        got = mz_after_n(torch.tensor(dt, dtype=torch.float64), torch.tensor(n), TR, T1, FA)
        assert float(got) == pytest.approx(_mz(dt, n), rel=1e-12)
    m_ss, q = steady_state(TR, T1, A)
    assert (m_ss, q) == pytest.approx((M_SS, E1 * math.cos(A)))
    # A hop with dt' = TR is the steady state at every n (no history).
    assert _gain(TR, n) == pytest.approx(1.0, rel=1e-12)


def _states(x, dz, n):
    """A + q^n B: the stack moved by dz with the first-excitation excess decaying over n lines."""
    max_hop = 2
    t0 = line_times_centre(4, 8)
    ex = excess_gain(intervals_at(MS, 4, 8, TR, range(-max_hop, max_hop + 1), t0), TR, T1, FA)
    ex[max_hop] = 0.0
    a = history_multislice(x, dz, torch.ones_like(ex[..., 0]), max_hop)
    b = history_multislice(x, dz, ex[..., 0], max_hop)
    return a + steady_state(TR, T1, A)[1] ** n * b


def line_times_centre(d, h):
    from spectramr.infrastructure.physics.degradations.blocks.acquisition_timing import line_times

    return line_times(MS, d, h, tr_s=TR).double()[:, h // 2 : h // 2 + 1]


@pytest.mark.parametrize("n", [0, 1, 5])
def test_a_spin_moved_one_pitch_follows_the_transient(n):
    x = torch.arange(1.0, 5.0).view(1, 1, 4, 1, 1).to(torch.complex64).expand(1, 1, 4, 8, 8)
    out = _states(x, torch.tensor([1.0]), n)
    # Slice 1 receives slice 0's spins 2/4 TR after they were excited, slice 2 receives slice 1's
    # 3/4 TR after, slice 3 receives slice 2's 2/4 TR after (interleaved ranks 0, 2, 1, 3).
    for k, src, dt in ((1, 1.0, 0.5 * TR), (2, 2.0, 0.75 * TR), (3, 3.0, 0.5 * TR)):
        assert out[0, 0, k, 0, 0].real == pytest.approx(src * _gain(dt, n), rel=1e-5)
    # Slice 0's source is outside the stack: never excited, M_0 = 1 (inflow).
    assert out[0, 0, 0, 0, 0].real == pytest.approx(1.0 * _gain(math.inf, n), rel=1e-5)
    assert _gain(math.inf, 0) > _gain(math.inf, 1) > _gain(math.inf, 5) > 1.0


def test_a_half_pitch_shift_mixes_the_two_source_slices():
    x = torch.arange(1.0, 5.0).view(1, 1, 4, 1, 1).to(torch.complex64).expand(1, 1, 4, 8, 8)
    out = _states(x, torch.tensor([0.5]), 1)
    # Slice 2 at dz = 0.5: half from slice 2 itself (steady state), half from slice 1.
    assert out[0, 0, 2, 0, 0].real == pytest.approx(
        0.5 * 3.0 + 0.5 * 2.0 * _gain(0.75 * TR, 1), rel=1e-5
    )


def test_lines_since_hop_counts_the_shots_own_excitations():
    shots = shot_index(line_times(MS, 4, 16, tr_s=TR), 4)
    n = lines_since_hop(line_times(MS, 4, 16, tr_s=TR).double(), shots, True)
    # Interleaved: each slice fires once per ky, so n runs 0, 1, 2, 3 within each 4-line shot.
    assert n[0].tolist() == [0, 1, 2, 3] * 4
    volume = replace(MS, mode=M.VOLUME_3D)
    t3 = line_times(volume, 4, 16, tr_s=TR)
    s3 = shot_index(t3, 8)
    n3 = lines_since_hop(t3.double(), s3, False)
    assert n3.flatten()[torch.argsort(t3.flatten())].tolist() == [0, 1, 2, 3, 4, 5, 6, 7] * 8


def test_zero_displacement_is_the_input_and_the_source_sign_follows_dz():
    x = torch.randn(2, 1, 4, 8, 8, dtype=torch.complex64)
    dt = hop_intervals(MS, 4, 8, TR, range(-2, 3))
    table = 1.0 + excess_gain(dt, TR, T1, FA)  # first-excitation gain per source offset
    torch.testing.assert_close(
        history_multislice(x, torch.zeros(2), torch.ones_like(table), 2), x, rtol=0, atol=1e-5
    )
    up = history_multislice(x, torch.tensor([-1.0, 1.0]), table, 2)
    assert up[1, 0, 1, 0, 0].abs() / x[1, 0, 0, 0, 0].abs() == pytest.approx(
        _gain(float(hop_intervals(MS, 4, 8, TR, (-1,))[0, 1]), 0), rel=1e-5
    )
    # dz = -1 sources slice k + 1: slice 0 of sample 0 holds slice 1's content.
    assert up[0, 0, 0, 0, 0].abs() / x[0, 0, 1, 0, 0].abs() == pytest.approx(
        _gain(float(hop_intervals(MS, 4, 8, TR, (1,))[0, 0]), 0), rel=1e-5
    )


def test_3d_only_the_edge_voxels_sourced_outside_the_slab_carry_the_excess():
    x = torch.ones(1, 1, 8, 4, 4, dtype=torch.complex64)
    a_in = float(excess_gain(torch.tensor(math.inf, dtype=torch.float64), TR, T1, FA))
    assert a_in == pytest.approx(1 / M_SS - 1)
    plain = history_volume(x, torch.tensor([1.0]), None)
    torch.testing.assert_close(plain, x, rtol=0, atol=1e-6)
    b = history_volume(x, torch.tensor([1.0]), a_in)
    assert b[0, 0, 0].real.mean() == pytest.approx(a_in, rel=1e-5)
    assert float(b[0, 0, 1:].abs().max()) < 1e-6
    half = history_volume(x, torch.tensor([0.5]), a_in)
    assert half[0, 0, 0].real.mean() == pytest.approx(0.5 * a_in, rel=1e-5)
    other = history_volume(x, torch.tensor([-1.0]), a_in)
    assert other[0, 0, -1].real.mean() == pytest.approx(a_in, rel=1e-5)
    assert float(other[0, 0, :-1].abs().max()) < 1e-6


@pytest.mark.parametrize("mode", [M.MULTISLICE_2D, M.VOLUME_3D])
def test_the_effect_starts_continuously_from_zero(mode):
    geom = replace(MS, mode=mode)
    x = torch.randn(1, 1, 6, 16, 16, dtype=torch.complex64)
    full = (apply_degradation(NAME, x, 1.0, seed=2, geometry=geom) - x).abs().mean()
    tiny = (apply_degradation(NAME, x, 1e-4, seed=2, geometry=geom) - x).abs().mean()
    assert 0 <= float(tiny) < 1e-3 * float(full)


def test_declared_peak_displacement_bounds_the_effect():
    # theta = 0.2 moves at most 1 mm = 0.2 pitch: slices far from the stack edge still hold
    # mostly their own content.
    x = torch.zeros(1, 1, 9, 16, 16, dtype=torch.complex64)
    x[:, :, 4] = 1.0
    out = apply_degradation(NAME, x, 0.2, seed=2, geometry=MS)
    assert float(out[0, 0, 4].abs().mean()) > 0.5
    assert float(out[0, 0, 7].abs().max()) < 1e-6 and float(out[0, 0, 1].abs().max()) < 1e-6


def test_profile_tr_and_flip_angle_are_forwarded():
    x = torch.randn(1, 1, 6, 8, 8, dtype=torch.complex64)
    prof = replace(FASTMRI_3T, geometry=MS, tr_ms=20.0, fa_deg=10.0)
    by_profile = apply_degradation(NAME, x, 0.8, seed=1, profile=prof)
    torch.testing.assert_close(
        by_profile, apply_degradation(NAME, x, 0.8, seed=1, geometry=MS, tr_ms=20.0, fa_deg=10.0)
    )
    other = apply_degradation(NAME, x, 0.8, seed=1, geometry=MS)
    assert not torch.allclose(by_profile, other)
    with pytest.raises(ValueError, match="fa_deg"):
        apply_degradation(NAME, x, 0.8, profile=prof, fa_deg=20.0)


@pytest.mark.parametrize(
    "kw", [{"fa_deg": 0.0}, {"fa_deg": 200.0}, {"tr_ms": -1.0}, {"t1_ms": 0.0}]
)
def test_bad_sequence_values_raise(kw):
    x = torch.ones(1, 1, 4, 8, 8, dtype=torch.complex64)
    with pytest.raises(ValueError, match=r"fa_deg|tr_ms|t1_ms"):
        apply_degradation(NAME, x, 0.5, geometry=MS, **kw)


def test_an_unknown_voxel_size_means_1_mm_voxels():
    x = torch.randn(1, 1, 4, 8, 8, dtype=torch.complex64)
    unknown = AcquisitionGeometry(mode=M.MULTISLICE_2D, voxel_mm=None)
    unit = AcquisitionGeometry(mode=M.MULTISLICE_2D, voxel_mm=(1.0, 1.0, 1.0))
    torch.testing.assert_close(
        apply_degradation(NAME, x, 0.5, seed=1, geometry=unknown),
        apply_degradation(NAME, x, 0.5, seed=1, geometry=unit),
    )


def _shot_case(order, d, h, n_shots):
    geom = replace(MS, slice_order=order)
    times = line_times(geom, d, h, tr_s=TR).double()
    shots = shot_index(times.float(), n_shots)
    first, start = shot_times(times, shots)
    return geom, times, shots, first, start


def test_interleaved_dt_runs_from_the_hop_not_from_the_first_new_excitation():
    # 4 slices x 8 lines in 4 shots: shot p is ky = 2p, 2p+1 of every slice, hop at 2p TR. Slice 3
    # (rank 3) first fires at (2p + 3/4) TR; its source, slice 2 (rank 1), fired at (2p + 1/4) TR,
    # after the hop and so never reached the moved spin. The spin's last excitation was slice 2's
    # at (2p - 3/4) TR: dt' = 3/2 TR (a cut at the first new excitation would give 1/2 TR).
    geom, _, _, first, start = _shot_case(SliceOrder.INTERLEAVED, 4, 8, 4)
    assert start.tolist() == pytest.approx([0.0, 2 * TR, 4 * TR, 6 * TR])
    dt = intervals_at(geom, 4, 8, TR, (-1,), first, cut=start)
    for p in (1, 2, 3):
        assert dt[0, 3, p] == pytest.approx(1.5 * TR, rel=1e-5)
        assert intervals_at(geom, 4, 8, TR, (-1,), first)[0, 3, p] == pytest.approx(
            0.5 * TR, rel=1e-5
        )
    # Slice 2 (rank 1) fires after its source slice 1 (rank 2) of the previous TR: unchanged.
    assert dt[0, 2, 1] == pytest.approx(0.75 * TR, rel=1e-5)


def test_sequential_dt_runs_from_the_last_excitation_before_the_hop():
    # 3 slices x 8 lines in 4 shots of 6 lines: shot 1 starts at 6 TR inside slice 0's block and
    # holds slice 1's lines 0-3 (8..11 TR). Slice 1's source (slice 0) last fired at 5 TR before
    # the hop, so dt' = 3 TR; slice 0's lines 6 and 7 (after the hop) never reached the spin.
    geom, times, shots, first, start = _shot_case(SliceOrder.SEQUENTIAL, 3, 8, 4)
    assert start[1] == pytest.approx(6 * TR) and first[1, 1] == pytest.approx(8 * TR)
    dt = intervals_at(geom, 3, 8, TR, (-1,), first, cut=start)
    assert dt[0, 1, 1] == pytest.approx(3 * TR, rel=1e-5)
    assert intervals_at(geom, 3, 8, TR, (-1,), first)[0, 1, 1] == pytest.approx(TR, rel=1e-5)
    # Slice 1 spans shots 1 and 2: n restarts at the shot boundary (lines 0-3, then 4-7).
    n = lines_since_hop(times, shots, True)
    assert n[1].tolist() == [0, 1, 2, 3, 0, 1, 2, 3]
