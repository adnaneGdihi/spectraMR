"""Low-pass circle / high-pass residual split used by the spectral cascade."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.infrastructure.physics.spectral_cascade_bands import (
    low_pass_circle_mask,
    split_low_high,
)


@pytest.mark.parametrize("size", [(16, 16), (15, 17)])
def test_split_is_exact_real_and_orthogonal(size: tuple[int, int]) -> None:
    x = torch.rand(2, 1, *size, generator=torch.Generator().manual_seed(0))
    low, high = split_low_high(x, low_pass_circle_mask(*size, cutoff=3))
    assert not low.is_complex()
    assert torch.allclose(low + high, x, atol=1e-6)
    # Hermitian-symmetric mask: the low band of a real image carries no imaginary part.
    assert abs(float((low * high).sum())) < 1e-4
    k_high = fft2c(torch.complex(high, torch.zeros_like(high)))
    centre = (size[0] // 2, size[1] // 2)
    assert k_high[..., centre[0], centre[1]].abs().max() < 1e-5


def test_mask_is_a_centred_disc() -> None:
    mask = low_pass_circle_mask(16, 16, cutoff=2)
    assert mask[8, 8] == 1 and mask[8, 10] == 1 and mask[8, 11] == 0
    assert int(mask.sum()) == 13
    assert torch.equal(mask, mask.flip(0).roll(1, 0))  # symmetric about index N // 2


def test_cutoff_at_nyquist_raises() -> None:
    with pytest.raises(ValueError, match="cutoff_frequency"):
        low_pass_circle_mask(16, 16, cutoff=8)
