"""ComplexChannelGain: one real gain per complex channel, so phase is preserved."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.blocks.complex_channel_gain import ComplexChannelGain

pytestmark = pytest.mark.unit


def _rotate(z: torch.Tensor, theta: float) -> torch.Tensor:
    """Multiply an interleaved ``[R1, I1, ...]`` tensor by ``e^{i theta}``."""
    cos, sin = torch.cos(torch.tensor(theta)), torch.sin(torch.tensor(theta))
    real, imag = z[:, 0::2], z[:, 1::2]
    out = torch.empty_like(z)
    out[:, 0::2] = cos * real - sin * imag
    out[:, 1::2] = sin * real + cos * imag
    return out


def _trained(gain: ComplexChannelGain) -> ComplexChannelGain:
    """Non-zero projection weights, as after training; zero-init is trivially equivariant."""
    torch.manual_seed(1)
    with torch.no_grad():
        gain.proj[1].weight.normal_()
        gain.proj[1].bias.normal_()
    return gain


def test_the_gain_commutes_with_a_global_phase_rotation() -> None:
    gain = _trained(ComplexChannelGain(16, 8))
    x, emb = torch.randn(2, 8, 6, 6), torch.randn(2, 16)
    with torch.no_grad():
        rotated_then_gained = gain(_rotate(x, 0.9), emb)
        gained_then_rotated = _rotate(gain(x, emb), 0.9)
    torch.testing.assert_close(rotated_then_gained, gained_then_rotated)


def test_a_per_real_channel_gain_would_rotate_phase() -> None:
    """PLANTED: the retired per-real-channel FiLM, which this block replaces, fails the
    same check, so the check above can tell the two apart."""
    torch.manual_seed(2)
    x = torch.randn(1, 8, 4, 4)
    per_real = 1.0 + torch.randn(1, 8, 1, 1)
    deviation = (_rotate(x, 0.9) * per_real - _rotate(x * per_real, 0.9)).abs().max()
    assert float(deviation) > 1e-3


def test_re_and_im_of_one_coefficient_share_a_factor() -> None:
    gain = _trained(ComplexChannelGain(4, 6))
    factor = gain.gain(torch.randn(3, 4))
    assert factor.shape == (3, 6)
    torch.testing.assert_close(factor[:, 0::2], factor[:, 1::2])


def test_zero_init_starts_every_block_at_identity() -> None:
    gain = ComplexChannelGain(16, 8)
    x = torch.randn(2, 8, 5, 5)
    assert torch.equal(gain(x, torch.randn(2, 16)), x)


def test_no_embedding_is_the_identity() -> None:
    gain = _trained(ComplexChannelGain(16, 8))
    x = torch.randn(1, 8, 3, 3)
    assert gain(x, None) is x


def test_a_complex_dtype_map_takes_one_gain_per_channel() -> None:
    gain = _trained(ComplexChannelGain(4, 3, interleaved=False))
    z = torch.randn(2, 3, 4, 4, dtype=torch.complex64)
    emb = torch.randn(2, 4)
    out = gain(z, emb)
    assert out.dtype == torch.complex64
    phase = torch.exp(torch.tensor(0.4j))
    torch.testing.assert_close(gain(z * phase, emb), out * phase)


def test_an_odd_interleaved_width_raises() -> None:
    with pytest.raises(ValueError, match="even channel count"):
        ComplexChannelGain(8, 5)


def test_a_width_mismatch_raises() -> None:
    gain = ComplexChannelGain(8, 4)
    with pytest.raises(ValueError, match="built for 4 channels"):
        gain(torch.randn(1, 6, 2, 2), torch.randn(1, 8))
