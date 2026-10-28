"""A 3-D per-slice-exact declaration must commute with the FFT along kz."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.degradations import (
    DEGRADATION_REGISTRY,
    AcquisitionGeometry,
    AcquisitionMode,
    Realisation,
)
from spectramr.infrastructure.physics.fft_ops import fftnc, ifftnc

M = AcquisitionMode
GEOM = AcquisitionGeometry(mode=M.VOLUME_3D, voxel_mm=None)
EXACT_3D = [
    name
    for name, spec in DEGRADATION_REGISTRY.items()
    if spec.realisation.get(M.VOLUME_3D) is Realisation.PER_SLICE_EXACT
]


def test_exact_3d_set_is_the_declared_one() -> None:
    assert set(EXACT_3D) == {
        "radial_undersamp",
        "nyquist_ghost",
        "t2star_blur",
        "chemical_shift",
        "stimulated_echo",
    }


@pytest.mark.parametrize("name", EXACT_3D)
def test_commutes_with_kz_fft(name: str) -> None:
    g = torch.Generator().manual_seed(11)
    shape = (1, 2, 6, 16, 16)
    x = torch.complex(torch.randn(shape, generator=g), torch.randn(shape, generator=g))
    impl = DEGRADATION_REGISTRY[name].implementations[M.VOLUME_3D]
    out = impl(x, 0.6, 5, GEOM)
    via_kz = ifftnc(impl(fftnc(x, dims=(-3,)), 0.6, 5, GEOM), dims=(-3,))
    scale = x.abs().max().item()
    torch.testing.assert_close(out, via_kz, rtol=1e-5, atol=1e-5 * scale)
