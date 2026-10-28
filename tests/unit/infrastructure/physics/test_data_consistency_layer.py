"""Tests for ``MaskedReplacementDataConsistency``.

Targets ``spectramr.infrastructure.physics.data_consistency_layer``. The
classic hard-DC layer: at sampled k-space locations, the network
guess is replaced with the acquired measurement.

Categories:

- Unit: complex input round-trip preserves shape; mask=0 → output equals
  iFFT(FFT(input)) (identity); mask=1 → output equals iFFT(measured)
- Property: idempotence — applying DC twice == once
- Property: 2-channel real input also supported
- Edge: odd C-channel input fails loud (not 2-channel or complex)
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.data_consistency_layer import MaskedReplacementDataConsistency
from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_default_construction() -> None:
    """Default ``noise_robust=False``."""
    dc = MaskedReplacementDataConsistency()
    assert dc.noise_robust is False


# ---------------------------------------------------------------------------
# Complex input
# ---------------------------------------------------------------------------


def test_complex_input_zero_mask_returns_iFFTFFT_input() -> None:
    """With mask=0, output equals iFFT(FFT(input)) (no DC enforced)."""
    dc = MaskedReplacementDataConsistency()
    img = torch.complex(torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8))
    measured = torch.complex(torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8))
    mask = torch.zeros(1, 1, 8, 8)
    out = dc(img, measured, mask)
    expected = ifft2c(fft2c(img))
    assert torch.allclose(out, expected, atol=1e-5)


def test_complex_input_full_mask_returns_iFFT_measured() -> None:
    """With mask=1, output equals iFFT(measured) (full overwrite)."""
    dc = MaskedReplacementDataConsistency()
    img = torch.complex(torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8))
    measured = torch.complex(torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8))
    mask = torch.ones(1, 1, 8, 8)
    out = dc(img, measured, mask)
    expected = ifft2c(measured)
    assert torch.allclose(out, expected, atol=1e-5)


def test_complex_input_preserves_shape() -> None:
    """Output shape matches input shape."""
    dc = MaskedReplacementDataConsistency()
    img = torch.complex(torch.randn(2, 1, 16, 16), torch.randn(2, 1, 16, 16))
    measured = torch.complex(torch.randn(2, 1, 16, 16), torch.randn(2, 1, 16, 16))
    mask = torch.zeros(2, 1, 16, 16)
    out = dc(img, measured, mask)
    assert out.shape == img.shape


def test_dc_is_idempotent() -> None:
    """Applying DC twice gives the same result as applying it once."""
    dc = MaskedReplacementDataConsistency()
    img = torch.complex(torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8))
    measured = torch.complex(torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8))
    mask = torch.zeros(1, 1, 8, 8)
    mask[..., :4, :] = 1.0  # half-mask
    once = dc(img, measured, mask)
    twice = dc(once, measured, mask)
    assert torch.allclose(once, twice, atol=1e-4)


def test_bool_mask_accepted() -> None:
    """Boolean mask is auto-converted to float."""
    dc = MaskedReplacementDataConsistency()
    img = torch.complex(torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8))
    measured = torch.complex(torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8))
    mask = torch.ones(1, 1, 8, 8, dtype=torch.bool)
    out = dc(img, measured, mask)
    assert torch.is_complex(out)


# ---------------------------------------------------------------------------
# 2-channel real input
# ---------------------------------------------------------------------------


def test_real_2channel_input_preserves_shape() -> None:
    """``[B, 2, H, W]`` real input → ``[B, 2, H, W]`` output."""
    dc = MaskedReplacementDataConsistency()
    img = torch.randn(1, 2, 8, 8)
    measured = torch.randn(1, 2, 8, 8)
    mask = torch.zeros(1, 1, 8, 8)
    out = dc(img, measured, mask)
    assert out.shape == img.shape


# ---------------------------------------------------------------------------
# Edge: odd channels
# ---------------------------------------------------------------------------


def test_odd_channel_count_raises() -> None:
    """Odd C (not complex, not 2-channel pairs) → ``ValueError``."""
    dc = MaskedReplacementDataConsistency()
    img = torch.randn(1, 3, 8, 8)
    measured = torch.randn(1, 3, 8, 8)
    mask = torch.zeros(1, 1, 8, 8)
    with pytest.raises(ValueError, match="2-channel"):
        dc(img, measured, mask)


# ---------------------------------------------------------------------------
# Sanity-shape matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape", [(1, 1, 8, 8), (2, 1, 16, 16), (1, 1, 32, 16)],
    ids=lambda s: "x".join(map(str, s)),
)
def test_complex_dc_shape_matrix(shape: tuple[int, ...]) -> None:
    """Complex DC handles the parametrised shape matrix."""
    dc = MaskedReplacementDataConsistency()
    img = torch.complex(torch.randn(*shape), torch.randn(*shape))
    measured = torch.complex(torch.randn(*shape), torch.randn(*shape))
    mask = torch.zeros(*shape)
    out = dc(img, measured, mask)
    assert out.shape == shape
    assert torch.is_complex(out)


# ---------------------------------------------------------------------------
# The shape diagnostic must not intercept checkpoint control flow
#
# ``torch.utils.checkpoint`` ends a non-reentrant recompute by raising
# ``_StopRecomputationError`` through the user code it is re-running. A
# ``except Exception`` around the blend caught that and reported a fabricated
# [DC LAYER CRASH] -- five per step on the 2026-09-17 diff_varnet run, which
# completed normally. Each test below plants one shape the handler must
# distinguish (non-negotiable 15).
# ---------------------------------------------------------------------------


def test_checkpoint_stop_signal_passes_through_unlogged(caplog, monkeypatch) -> None:
    """The checkpoint early-stop is re-raised without a [DC LAYER CRASH] record."""
    from torch.utils.checkpoint import _StopRecomputationError

    def _stop(*_args, **_kwargs):
        raise _StopRecomputationError

    # The blend's first operand op: raising here stands in for autograd throwing
    # the early-stop into the recompute at exactly this point.
    monkeypatch.setattr(torch.Tensor, "__rsub__", _stop)

    dc = MaskedReplacementDataConsistency()
    with caplog.at_level("ERROR"), pytest.raises(_StopRecomputationError):
        dc(torch.randn(1, 2, 8, 8), torch.randn(1, 2, 8, 8), torch.ones(1, 1, 8, 8))

    assert "DC LAYER CRASH" not in caplog.text


def test_a_real_broadcast_failure_is_still_reported(caplog) -> None:
    """A mask on the wrong grid still raises AND still logs the three shapes."""
    dc = MaskedReplacementDataConsistency()
    img = torch.randn(1, 2, 8, 8)
    measured = torch.randn(1, 2, 8, 8)
    mask = torch.ones(1, 1, 4, 4)  # right channel count, wrong spatial grid

    with caplog.at_level("ERROR"), pytest.raises(RuntimeError):
        dc(img, measured, mask)

    assert "DC LAYER CRASH" in caplog.text
    assert "torch.Size([1, 1, 4, 4])" in caplog.text


def test_the_report_carries_the_exception_not_only_the_shapes(caplog) -> None:
    """``exc_info`` is attached, so the cluster log says what actually failed."""
    dc = MaskedReplacementDataConsistency()
    with caplog.at_level("ERROR"), pytest.raises(RuntimeError):
        dc(torch.randn(1, 2, 8, 8), torch.randn(1, 2, 8, 8), torch.ones(1, 1, 4, 4))

    crash = [r for r in caplog.records if "DC LAYER CRASH" in r.getMessage()]
    assert len(crash) == 1
    assert crash[0].exc_info is not None
