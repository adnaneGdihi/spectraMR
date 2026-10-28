"""The linear complex view of a reverse-loop estimate round-trips in both domains."""

from __future__ import annotations

import pytest
import torch

from spectramr.data.transforms.normalization import compress_kspace_log
from spectramr.models.diffusion.reverse_domain import linear_kspace, stacked_kspace

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("log_scaled", [False, True])
def test_the_round_trip_is_the_identity(log_scaled) -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 6, 8, 8)
    x = compress_kspace_log(x, channel_dim=1) if log_scaled else x
    z = linear_kspace(x, log_scaled=log_scaled)
    assert z.is_complex() and z.shape == (2, 3, 8, 8)
    assert torch.allclose(stacked_kspace(z, x, log_scaled=log_scaled), x, atol=1e-5)


def test_the_pairs_are_interleaved_real_then_imaginary() -> None:
    x = torch.zeros(1, 4, 2, 2)
    x[:, 0], x[:, 1], x[:, 2], x[:, 3] = 1.0, 2.0, 3.0, 4.0
    z = linear_kspace(x, log_scaled=False)
    assert torch.equal(z[:, 0], torch.full((1, 2, 2), 1 + 2j))
    assert torch.equal(z[:, 1], torch.full((1, 2, 2), 3 + 4j))


def test_the_log_view_is_linear_in_magnitude() -> None:
    """A compressed input decompresses before any physics sees it."""
    x = torch.zeros(1, 2, 1, 1)
    x[:, 0] = 3.0
    z = linear_kspace(compress_kspace_log(x, channel_dim=1), log_scaled=True)
    assert torch.allclose(z.abs(), torch.full_like(z.abs(), 3.0), atol=1e-5)


def test_a_complex_estimate_passes_through() -> None:
    z = torch.randn(1, 2, 4, 4, dtype=torch.complex64)
    assert torch.equal(linear_kspace(z, log_scaled=False), z)
    assert torch.equal(stacked_kspace(z, z, log_scaled=False), z)
