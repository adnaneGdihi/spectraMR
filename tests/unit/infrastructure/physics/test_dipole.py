"""The centred 3-D dipole kernel against its closed form and the uncentred owner's sign."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.dipole import dipole_kernel_centred, get_dipole_kernel_3d
from spectramr.infrastructure.physics.fft_ops import centred_freqs


def _roll_to_centre(k: torch.Tensor) -> torch.Tensor:
    # fftfreq order -> DC at N // 2, the centred_freqs layout.
    return torch.roll(k, shifts=tuple(n // 2 for n in k.shape), dims=(0, 1, 2))


@pytest.mark.parametrize("shape", [(6, 8, 10), (5, 7, 9)])
@pytest.mark.parametrize("b0_dim", [-3, -2, -1])
def test_closed_form_and_dc(shape, b0_dim):
    kern = dipole_kernel_centred(shape, b0_dim)
    assert kern.shape == shape
    k = torch.meshgrid(*[centred_freqs(n) for n in shape], indexing="ij")
    k2 = sum(c**2 for c in k)
    assert float(kern[shape[0] // 2, shape[1] // 2, shape[2] // 2]) == 0.0
    nz = k2 > 0
    expected = 1.0 / 3.0 - k[b0_dim + 3] ** 2 / k2.clamp_min(1e-30)
    torch.testing.assert_close(kern[nz], expected[nz])


@pytest.mark.parametrize("shape", [(6, 8, 10), (5, 7, 9)])
def test_sign_matches_uncentred_owner(shape):
    ref = _roll_to_centre(get_dipole_kernel_3d(shape))
    torch.testing.assert_close(dipole_kernel_centred(shape, -3), ref, rtol=1e-5, atol=1e-6)


def test_voxel_mm_uses_physical_frequencies():
    shape, voxel = (6, 8, 10), (3.0, 1.0, 2.0)
    ref = _roll_to_centre(get_dipole_kernel_3d(shape, voxel))
    got = dipole_kernel_centred(shape, -3, voxel_mm=voxel)
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-6)


def test_bad_b0_dim_raises():
    with pytest.raises(ValueError, match="b0_dim"):
        dipole_kernel_centred((4, 4, 4), 0)
