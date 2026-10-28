"""SENSE aliasing plan, g-factor and unfolding: checked against k-space sampling and a hand inverse."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from spectramr.infrastructure.physics.coil_sensitivity import sense_gfactor_map
from spectramr.infrastructure.physics.degradations import AcquisitionMode
from spectramr.infrastructure.physics.degradations.catalogue.recon_sense import gfactor_noise_volume
from spectramr.infrastructure.physics.degradations.catalogue.sense_core import (
    alias_plan,
    fold_members,
    gfactor,
    pad_to_plan,
    scatter_members,
    seeded_smooth_csm,
    sense_unfold,
)
from spectramr.infrastructure.physics.fft_ops import fftnc, ifftnc

CPU = torch.device("cpu")


def _csm(coils, shape, voxel=None, seed=0, batch=1):
    gen = torch.Generator().manual_seed(seed)
    return seeded_smooth_csm(batch, coils, shape, voxel, generator=gen, device=CPU)


@pytest.mark.parametrize(("r_pe", "r_sl", "shift"), [(2, 1, 0), (4, 1, 0), (2, 2, 0), (2, 2, 1)])
def test_aliased_sets_partition_the_plane_on_the_dual_lattice(r_pe, r_sl, shift):
    d, h = 8, 8
    plan = alias_plan((d, h), r_pe, r_sl, shift, CPU)
    flat = plan.members.reshape(-1).sort().values
    assert torch.equal(flat, torch.arange(d * h))
    dr, hr = d // r_sl, h // r_pe
    for z0, y0 in [(0, 0), (1, 3 % hr), (dr - 1, hr - 1)]:
        want = {
            (z0 + q * dr) * h + (y0 + p * hr - q * (hr * shift // r_sl)) % h
            for p in range(r_pe)
            for q in range(r_sl)
        }
        assert set(plan.members[z0, y0].tolist()) == want


@pytest.mark.parametrize(("r_pe", "r_sl", "shift"), [(2, 1, 0), (4, 1, 0), (2, 2, 0), (2, 2, 1)])
def test_the_alias_sets_are_what_sampling_the_ky_kz_lattice_produces(r_pe, r_sl, shift):
    """ifft(M fft(x)) = (1/n) sum over the set, for M the sheared (ky, kz) lattice (Breuer 2006)."""
    d, h, w = 8, 8, 2
    g = torch.Generator().manual_seed(1)
    x = torch.complex(
        torch.randn(1, 1, d, h, w, generator=g), torch.randn(1, 1, d, h, w, generator=g)
    )
    kz = (torch.arange(d) - d // 2).view(d, 1)
    ky = (torch.arange(h) - h // 2).view(1, h)
    # Breuer's lattice: (ky, kz) = m1 (R_y, shift) + m2 (0, R_z): ky a multiple of R_y, kz - shift ky / R_y of R_z.
    on = (ky % r_pe == 0) & ((kz - shift * (ky // r_pe)) % r_sl == 0)
    sampled = ifftnc(fftnc(x, (-3, -2, -1)) * on.view(1, 1, d, h, 1), (-3, -2, -1))
    plan = alias_plan((d, h), r_pe, r_sl, shift, CPU)
    summed = fold_members(x, plan).sum(-1).sum(-1)  # [B, Dr, Hr, W]
    want = scatter_members(
        (summed / plan.n).unsqueeze(-1).expand(-1, -1, -1, -1, plan.n), plan
    )  # every member sees the set mean
    torch.testing.assert_close(sampled, want, rtol=0, atol=1e-5)


def test_caipi_shift_that_does_not_divide_raises():
    with pytest.raises(ValueError, match="CAIPI shift"):
        alias_plan((8, 6), 2, 2, 1, CPU)


@pytest.mark.parametrize("mode", [AcquisitionMode.MULTISLICE_2D, AcquisitionMode.VOLUME_3D])
def test_hand_checked_aliased_set_for_r_2x2_with_caipi_shift_1(mode):
    """Breuer 2D-CAIPIRINHA on D = H = 8: voxel (z, y) = (1, 1) aliases with the excited-together slice
    z + D/2 = 5 and, per partner, y + H/2 shifted by -q delta H / (R_y R_z) = -2 rows for q = 1:
    (z, y) in {(1, 1), (1, 5), (5, 7), (5, 3)}. The same set serves multislice (SMS) and 3-D."""
    from spectramr.infrastructure.physics.degradations import AcquisitionGeometry

    plan = alias_plan((8, 8), 2, 2, 1, CPU)
    assert set(plan.members[1, 1].tolist()) == {1 * 8 + 1, 1 * 8 + 5, 5 * 8 + 7, 5 * 8 + 3}
    geom = AcquisitionGeometry(mode=mode, voxel_mm=(1.0, 1.0, 1.0))
    x = torch.ones(1, 1, 8, 8, 4, dtype=torch.complex64)
    out = gfactor_noise_volume(x, 1.0, 0, geom, acceleration=(2.0, 2.0), n_coils=8)
    ref = gfactor_noise_volume(
        x,
        1.0,
        0,
        AcquisitionGeometry(mode=AcquisitionMode.VOLUME_3D, voxel_mm=(1.0, 1.0, 1.0)),
        acceleration=(2.0, 2.0),
        n_coils=8,
    )
    assert torch.equal(out, ref)


def test_padding_makes_the_plane_divisible_and_replicates_maps():
    plan = alias_plan((1, 33), 2, 1, 0, CPU)
    assert plan.shape == (1, 34)
    csm = _csm(3, (1, 33, 4))
    padded = pad_to_plan(csm, plan, "replicate")
    assert padded.shape[-2] == 34 and torch.equal(padded[..., 33, :], padded[..., 32, :])
    zero = pad_to_plan(csm, plan, "constant")
    assert zero[..., 33, :].abs().max() == 0


@pytest.mark.parametrize("coils", [4, 8])
def test_gfactor_matches_the_existing_2d_sense_gfactor_map(coils):
    csm = _csm(coils, (1, 32, 16))
    plan = alias_plan((1, 32), 2, 1, 0, CPU)
    g = gfactor(csm, plan, None)
    ref = sense_gfactor_map(csm[:, :, 0], 2)
    torch.testing.assert_close(g[:, :, 0], ref, rtol=5e-4, atol=1e-4)
    assert float(g.min()) >= 1.0 - 1e-4  # g >= 1 (Pruessmann 1999)


def test_orthogonal_maps_give_unit_g():
    """Coil 0 sees only the top half and coil 1 only the bottom half: A is diagonal, g = 1."""
    csm = torch.zeros(1, 2, 1, 8, 3, dtype=torch.complex64)
    csm[:, 0, :, :4], csm[:, 1, :, 4:] = 1.0, 2.0 * 1j
    g = gfactor(csm, alias_plan((1, 8), 2, 1, 0, CPU), None)
    torch.testing.assert_close(g, torch.ones_like(g), rtol=0, atol=1e-4)


def test_near_degenerate_pair_matches_a_hand_inverse_and_identical_maps_blow_up():
    def g_of(delta):
        e = np.array([[1.0, 1.0], [0.5j, 0.5j + delta]])  # coils x the two aliased voxels
        csm = torch.zeros(1, 2, 1, 2, 1, dtype=torch.complex64)
        csm[0, :, 0, :, 0] = torch.from_numpy(e).to(torch.complex64)
        got = gfactor(csm, alias_plan((1, 2), 2, 1, 0, CPU), None)[0, 0, 0, :, 0]
        a = e.conj().T @ e
        want = np.sqrt(np.diag(np.linalg.inv(a)).real * np.diag(a).real) if delta else None
        return got.numpy(), want

    got, want = g_of(0.3)
    np.testing.assert_allclose(got, want, rtol=1e-4)
    assert want.min() > 2.0  # a genuinely amplified case
    blown, _ = g_of(0.0)
    assert (
        blown.min() > 100.0
    )  # identical sensitivities: g -> infinity, capped by the diagonal load


def test_noise_covariance_enters_g_through_psi_inverse():
    csm = _csm(4, (1, 16, 4))
    plan = alias_plan((1, 16), 2, 1, 0, CPU)
    psi = torch.diag(torch.tensor([1.0, 4.0, 1.0, 9.0], dtype=torch.float64))
    got = gfactor(csm, plan, torch.linalg.inv(psi))
    ref = sense_gfactor_map(csm[:, :, 0], 2, noise_cov=psi.to(torch.complex64))
    torch.testing.assert_close(got[:, :, 0], ref, rtol=0, atol=1e-3)


@pytest.mark.parametrize(("r_pe", "r_sl", "shift", "d"), [(2, 1, 0, 1), (3, 1, 0, 1), (2, 2, 1, 8)])
def test_unfolding_the_true_maps_recovers_the_object(r_pe, r_sl, shift, d):
    h, w = 12, 4
    csm = _csm(8, (d, h, w))
    g = torch.Generator().manual_seed(5)
    m = torch.complex(
        torch.randn(1, 1, d, h, w, generator=g), torch.randn(1, 1, d, h, w, generator=g)
    )
    plan = alias_plan((d, h), r_pe, r_sl, shift, CPU)
    aliased = fold_members(csm * m, plan).sum(-1)
    torch.testing.assert_close(sense_unfold(aliased, csm, plan, None), m, rtol=0, atol=1e-3)


def test_synthetic_maps_are_unit_rss_seeded_smooth_and_batch_independent():
    a = _csm(8, (4, 16, 16), (5.0, 1.0, 1.0), seed=3)
    torch.testing.assert_close(a.abs().pow(2).sum(1), torch.ones(1, 4, 16, 16), rtol=0, atol=1e-5)
    assert torch.equal(a, _csm(8, (4, 16, 16), (5.0, 1.0, 1.0), seed=3))
    assert not torch.equal(a, _csm(8, (4, 16, 16), (5.0, 1.0, 1.0), seed=4))
    assert float((a[..., 1:, :] - a[..., :-1, :]).abs().max()) < 0.5  # no branch cut
    assert torch.equal(
        _csm(8, (4, 16, 16), None, seed=3, batch=2)[:1], _csm(8, (4, 16, 16), None, seed=3)
    )
