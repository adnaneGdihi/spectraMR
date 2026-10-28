"""flow_ghosts: closed-form ghost positions and amplitudes."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from spectramr.infrastructure.physics.degradations import (
    AcquisitionGeometry,
    LineOrder,
    SliceOrder,
    apply_degradation,
)
from spectramr.infrastructure.physics.scanner_presets import FASTMRI_3T

from .flow_helpers import GEOMS, VOXEL, M, _apply, _phantom, _rms  # noqa: F401

# ---- flow_ghosts: closed form ---------------------------------------------------------------


def _dot(shape, pos, value=2.0):
    """A unit background with one bright vessel pixel at ``pos`` and the one-hot map for it."""
    x = torch.ones(shape, dtype=torch.complex64)
    vessel = torch.zeros(shape)
    vessel[(..., *pos)] = 1.0
    x[(..., *pos)] = value
    return x, vessel


def _ghost_positions(diff, tol=1e-5):
    return {tuple(int(i) for i in p) for p in (diff.abs() > tol).nonzero()[:, 2:]}


def test_ghosts_of_one_pixel_sit_at_the_predicted_offset_2d():
    # f = TR / T_card = 1/16 cycle per line; N_pe = 64 -> ghosts at +/- 4 px along PE, nowhere else.
    x, vessel = _dot((1, 1, 64, 16), (30, 7))
    y = apply_degradation(
        "flow_ghosts", x, 0.5, seed=3, vessel_map=vessel, tr_ms=50.0, t_card_ms=800.0
    )
    diff = y - x
    assert _ghost_positions(diff) == {(26, 7), (34, 7)}
    a = 0.8 * 0.5
    up, down = diff[0, 0, 34, 7], diff[0, 0, 26, 7]
    assert up.abs().item() == pytest.approx(a * 2.0 / 2, rel=1e-4)
    assert down.abs().item() == pytest.approx(a * 2.0 / 2, rel=1e-4)
    # cos = (e^{i.} + e^{-i.}) / 2: the pair carries opposite phases, so their product is real.
    assert (up * down).real.item() == pytest.approx((a * 2.0 / 2) ** 2, rel=1e-3)
    assert abs((up * down).imag.item()) < 1e-4


def test_ghost_energy_is_the_modulation_energy():
    x, vessel = _dot((1, 1, 64, 16), (30, 7))
    y = apply_degradation(
        "flow_ghosts", x, 1.0, seed=3, vessel_map=vessel, tr_ms=50.0, t_card_ms=800.0
    )
    assert float((y - x).abs().pow(2).sum()) == pytest.approx(2 * (0.8 * 2.0 / 2) ** 2, rel=1e-3)


def test_ghost_offset_follows_the_period_and_tr():
    x, vessel = _dot((1, 1, 64, 16), (30, 7))
    kw = {"vessel_map": vessel, "seed": 3}
    y = apply_degradation("flow_ghosts", x, 0.5, tr_ms=100.0, t_card_ms=800.0, **kw)
    assert _ghost_positions(y - x) == {(22, 7), (38, 7)}  # f = 1/8 -> 8 px


def test_ghosts_follow_the_profile_tr_and_refuse_a_clashing_kwarg():
    x, vessel = _dot((1, 1, 64, 16), (30, 7))
    prof = replace(FASTMRI_3T, tr_ms=50.0, te_ms=10.0)
    y = apply_degradation(
        "flow_ghosts", x, 0.5, seed=3, profile=prof, vessel_map=vessel, t_card_ms=800.0
    )
    assert _ghost_positions(y - x) == {(26, 7), (34, 7)}
    with pytest.raises(ValueError, match="tr_ms"):
        apply_degradation("flow_ghosts", x, 0.5, profile=prof, tr_ms=60.0)


def _volume_ghosts(line_order, tr_ms, t_card_ms):
    shape = (1, 1, 4, 16, 8)
    x, vessel = _dot(shape, (1, 6, 3))
    geom = AcquisitionGeometry(mode=M.VOLUME_3D, voxel_mm=VOXEL, line_order=line_order)
    y = apply_degradation(
        "flow_ghosts",
        x,
        0.5,
        seed=3,
        geometry=geom,
        vessel_map=vessel,
        tr_ms=tr_ms,
        t_card_ms=t_card_ms,
    )
    pos = {tuple(int(i) for i in p) for p in ((y - x).abs() > 1e-5).nonzero()[:, 2:]}
    return pos


def test_volume_ghosts_honour_line_order():
    # f = 1/4: linear t = (kz Ny + ky) TR shifts ky by 4 px and kz by Nz Ny f = 16 = 0 (mod 4);
    # kz-inner t = (ky Nz + kz) TR shifts kz by 1 px and ky by Ny Nz f = 16 = 0 (mod 16).
    assert _volume_ghosts(LineOrder.LINEAR, 50.0, 200.0) == {(1, 2, 3), (1, 10, 3)}
    assert _volume_ghosts(LineOrder.KZ_INNER, 50.0, 200.0) == {(0, 6, 3), (2, 6, 3)}


def test_multislice_ghosts_stay_in_their_slice_and_follow_slice_order():
    shape = (1, 1, 4, 16, 8)
    x, vessel = _dot(shape, (1, 6, 3))
    outs = {}
    for order in SliceOrder:
        geom = AcquisitionGeometry(mode=M.MULTISLICE_2D, voxel_mm=VOXEL, slice_order=order)
        outs[order] = apply_degradation(
            "flow_ghosts",
            x,
            0.5,
            seed=3,
            geometry=geom,
            vessel_map=vessel,
            tr_ms=50.0,
            t_card_ms=200.0,
        )
        pos = {tuple(int(i) for i in p) for p in ((outs[order] - x).abs() > 1e-5).nonzero()[:, 2:]}
        assert pos == {(1, 2, 3), (1, 10, 3)}
    # Sequential slices start (slice * H) lines later; interleaved ones a fraction of a TR.
    assert not torch.allclose(outs[SliceOrder.SEQUENTIAL], outs[SliceOrder.INTERLEAVED])


def test_ghosts_leave_pixels_outside_the_vessel_column_alone_with_a_seeded_vessel():
    x = torch.ones(1, 1, 64, 32, dtype=torch.complex64)
    y = apply_degradation("flow_ghosts", x, 1.0, seed=2, tr_ms=50.0, t_card_ms=800.0)
    # Ghosts of a vessel are replicas of the vessel only: columns it never touches are exact.
    changed_cols = ((y - x).abs() > 1e-6).any(dim=-2)[0, 0]
    assert 0 < int(changed_cols.sum()) < x.shape[-1]
