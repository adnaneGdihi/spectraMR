"""Tests for the spectral band-split loss.

Targets ``spectramr.models.losses.spectral_band_split_loss.SpectralBandSplitLoss``.

Regression: the loss used to pre-cast real inputs with ``.to(torch.complex64)``
before calling ``fft2c``. That defeats ``fft2c``'s ``_to_complex`` routing,
which interleaves an even-channel real/imag layout (``[B, 2, H, W]``) into a
single complex channel. Pre-casting instead FFT'd each of the two channels as a
separate real image. The fix passes tensors straight to ``fft2c``; this test
pins the invariant that the ``[B, 2, H, W]`` interleaved representation and the
equivalent explicit ``[B, 1, H, W]`` complex tensor yield the *same* loss.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.models.losses.spectral_band_split_loss import (  # noqa: E402
    SpectralBandSplitLoss,
)


def _interleave(c: torch.Tensor) -> torch.Tensor:
    """[B, 1, H, W] complex -> [B, 2, H, W] real, channel order [real, imag].

    This matches the interleaving ``fft_ops._to_complex`` expects for an
    even-channel real tensor.
    """
    return torch.cat([c.real, c.imag], dim=1)


def test_interleaved_real_matches_explicit_complex() -> None:
    """A 2-channel real/imag input must give the same loss as the equivalent
    complex tensor — i.e. ``fft2c`` interleaves it into one complex image."""
    torch.manual_seed(0)

    def _cimg() -> torch.Tensor:
        return torch.complex(torch.randn(1, 1, 16, 16), torch.randn(1, 1, 16, 16))

    pred_c, src_c, tgt_c = _cimg(), _cimg(), _cimg()

    loss_fn = SpectralBandSplitLoss(center_fraction=0.2, high_loss_type="l1")

    out_complex = loss_fn(pred_c, src_c, tgt_c)
    out_real2ch = loss_fn(_interleave(pred_c), _interleave(src_c), _interleave(tgt_c))

    for key in ("loss_low", "loss_high", "spectral_total"):
        assert out_real2ch[key].item() == pytest.approx(
            out_complex[key].item(), rel=1e-5, abs=1e-6
        ), f"{key}: 2-channel interleaved should match explicit complex"


def test_identical_prediction_source_has_zero_low_band() -> None:
    """When G(x) == x the low-frequency consistency residual is zero."""
    torch.manual_seed(0)
    img = torch.rand(1, 1, 16, 16)
    out = SpectralBandSplitLoss()(img, img, None)
    assert out["loss_low"].item() == pytest.approx(0.0, abs=1e-6)


def test_registered_under_canonical_name() -> None:
    from spectramr.models.losses.registry import list_available

    assert "spectral_band_split" in list_available()


def test_unknown_high_loss_type_raises() -> None:
    """``high_loss_type`` is validated; a typo no longer silently selects L1."""
    with pytest.raises(ValueError, match="high_loss_type"):
        SpectralBandSplitLoss(high_loss_type="mse")


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_center_fraction_outside_unit_interval_raises(bad: float) -> None:
    from spectramr.models.losses.spectral_band_split_loss import SpectralBandSplitLoss

    with pytest.raises(ValueError, match="center_fraction"):
        SpectralBandSplitLoss(center_fraction=bad)
