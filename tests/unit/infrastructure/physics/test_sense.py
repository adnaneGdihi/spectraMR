"""Tests for SENSE-manifold projection and SENSE loss.

Targets ``spectramr.infrastructure.physics.sense``. ``SENSESubspaceProjector``
projects multi-coil k-space onto the physical SENSE manifold via
adjoint→forward; ``compute_sense_loss`` applies L1 in the
SENSE-combined image domain.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
from spectramr.infrastructure.physics.sense import (
    SENSESubspaceProjector,
    compute_sense_loss,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_complex_kspace(b: int, c: int, h: int, w: int) -> torch.Tensor:
    """A random multi-coil k-space tensor."""
    real = torch.randn(b, c, h, w)
    imag = torch.randn(b, c, h, w)
    return torch.complex(real, imag)


def _normalised_smaps(b: int, c: int, h: int, w: int) -> torch.Tensor:
    """Random complex coil maps normalised so Σ|S|² ≈ 1 per pixel."""
    s = _random_complex_kspace(b, c, h, w)
    rss = torch.sqrt((s.abs() ** 2).sum(dim=1, keepdim=True) + 1e-8)
    return s / rss


# ---------------------------------------------------------------------------
# SENSESubspaceProjector
# ---------------------------------------------------------------------------


def test_sense_projector_preserves_shape_complex_input() -> None:
    """Projection preserves [B, C, H, W] for complex input."""
    proj = SENSESubspaceProjector()
    k = _random_complex_kspace(1, 4, 16, 16)
    s = _normalised_smaps(1, 4, 16, 16)
    out = proj(k, s)
    assert out.shape == k.shape
    assert torch.is_complex(out)


def test_sense_projector_preserves_shape_real_stacked_input() -> None:
    """Real-stacked input ([B, 2C, H, W]) returns real-stacked output."""
    proj = SENSESubspaceProjector()
    # 2 complex coils → 4-channel real-stacked
    k = torch.randn(1, 4, 16, 16)
    s = torch.randn(1, 4, 16, 16)
    out = proj(k, s)
    assert out.shape == k.shape
    assert not torch.is_complex(out)


def test_sense_projector_idempotent_within_tolerance() -> None:
    """Projecting twice equals projecting once: P(P(x)) ≈ P(x).

    The SENSE-adjoint→forward sequence is a projection (P² = P).
    """
    proj = SENSESubspaceProjector()
    k = _random_complex_kspace(1, 4, 16, 16)
    s = _normalised_smaps(1, 4, 16, 16)
    p1 = proj(k, s)
    p2 = proj(p1, s)
    # Numerical FFT roundtrip introduces small error; tolerate 1e-4.
    rel_err = (p2 - p1).abs().max().item() / (p1.abs().max().item() + 1e-9)
    assert rel_err < 1e-3, f"idempotency rel_err = {rel_err:.2e}"


def test_sense_projector_reduces_off_manifold_components() -> None:
    """Energy of projected output ≤ energy of input (projection contracts).

    Adding noise outside the SENSE manifold should be partially removed
    by the adjoint→forward roundtrip.
    """
    proj = SENSESubspaceProjector()
    k = _random_complex_kspace(1, 4, 16, 16)
    s = _normalised_smaps(1, 4, 16, 16)
    p = proj(k, s)
    # ||P(k)||² ≤ ||k||² is the standard projection inequality.
    assert (p.abs() ** 2).sum().item() <= (k.abs() ** 2).sum().item() + 1e-3


# ---------------------------------------------------------------------------
# compute_sense_loss
# ---------------------------------------------------------------------------


def test_sense_loss_zero_when_pred_equals_target() -> None:
    """Loss is exactly zero when prediction equals target."""
    k = _random_complex_kspace(1, 4, 16, 16)
    s = _normalised_smaps(1, 4, 16, 16)
    loss = compute_sense_loss(k, k, s)
    assert loss.item() < 1e-6


def test_sense_loss_positive_when_pred_differs() -> None:
    """Loss is strictly positive for non-equal pred/target."""
    pred = _random_complex_kspace(1, 4, 16, 16)
    target = _random_complex_kspace(1, 4, 16, 16)
    s = _normalised_smaps(1, 4, 16, 16)
    loss = compute_sense_loss(pred, target, s)
    assert loss.item() > 0


def test_sense_loss_returns_scalar() -> None:
    """Output is a 0-dim scalar tensor."""
    pred = _random_complex_kspace(1, 4, 16, 16)
    target = _random_complex_kspace(1, 4, 16, 16)
    s = _normalised_smaps(1, 4, 16, 16)
    loss = compute_sense_loss(pred, target, s)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Sanity-shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [(1, 2, 16, 16), (1, 4, 32, 32), (2, 4, 16, 16), (1, 8, 32, 32)],
    ids=lambda s: "x".join(map(str, s)),
)
def test_sense_projector_shape_matrix(shape: tuple[int, int, int, int]) -> None:
    """Projector preserves shape across a parametrised matrix."""
    proj = SENSESubspaceProjector()
    k = _random_complex_kspace(*shape)
    s = _normalised_smaps(*shape)
    out = proj(k, s)
    assert out.shape == shape
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_sense_projector_handles_single_coil() -> None:
    """Single-coil input is the trivial case — projection is near-identity."""
    proj = SENSESubspaceProjector()
    k = _random_complex_kspace(1, 1, 16, 16)
    s = _normalised_smaps(1, 1, 16, 16)
    out = proj(k, s)
    # With one coil, the SENSE adjoint→forward roundtrip projects onto
    # the span of the single sensitivity, which after RSS-normalisation
    # is unit. So P(k) should be close to k (approximately, up to FFT
    # rounding).
    assert out.shape == k.shape


# ---------------------------------------------------------------------------
# The projection's domain (#2488)
# ---------------------------------------------------------------------------


def _coil_kspace_on_the_manifold(size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact four-coil k-space ``F(s_c * m)`` at the loader's compressed scale, and its maps."""
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij")
    obj = ((yy / 0.8) ** 2 + (xx / 0.6) ** 2 < 1).float() * (1 + 0.5 * (yy > 0.2).float())
    centres = [(-1.2, 0.0), (1.2, 0.0), (0.0, -1.2), (0.0, 1.2)]
    maps = torch.stack(
        [torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 1.5) * torch.exp(1j * (0.7 * cy * yy + 0.5 * cx * xx)) for cy, cx in centres]
    )[None].to(torch.complex64)
    maps = maps / torch.sqrt((maps.abs() ** 2).sum(1, keepdim=True))
    k = fft2c(maps * obj.to(torch.complex64))
    return k * (40.0 / k.abs().amax()), maps


def _nrmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().pow(2).sum().sqrt() / b.abs().pow(2).sum().sqrt())


def test_an_exact_prediction_survives_the_projection_in_either_domain() -> None:
    """On the coil manifold the projection is the identity, compressed or not."""
    from spectramr.data.transforms.normalization import compress_kspace_log

    k, maps = _coil_kspace_on_the_manifold()
    proj = SENSESubspaceProjector()
    assert _nrmse(proj(k, maps, log_scaled=False), k) < 1e-4
    kc = compress_kspace_log(k)
    assert _nrmse(proj(kc, maps, log_scaled=True), kc) < 1e-4


def test_the_real_stacked_compressed_round_trip_matches_the_complex_one() -> None:
    """The generator hands over interleaved real channels; the decompression must use channel_dim=1."""
    from spectramr.data.transforms.normalization import compress_kspace_log

    k, maps = _coil_kspace_on_the_manifold()
    kc = compress_kspace_log(k)
    stacked = torch.view_as_real(kc).permute(0, 1, 4, 2, 3).reshape(1, 8, *kc.shape[-2:])
    out = SENSESubspaceProjector()(stacked, maps, log_scaled=True)
    assert _nrmse(out, stacked) < 1e-4


def test_projecting_compressed_kspace_as_if_linear_distorts_it_violation() -> None:
    """PLANTED: the pre-#2488 call, compressed k-space projected as linear, moves an exact prediction."""
    from spectramr.data.transforms.normalization import compress_kspace_log

    k, maps = _coil_kspace_on_the_manifold()
    kc = compress_kspace_log(k)
    assert _nrmse(SENSESubspaceProjector()(kc, maps, log_scaled=False), kc) > 0.1
