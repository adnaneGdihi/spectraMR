"""Tests for ``NullSpaceMarkerEraser``.

Targets ``spectramr.infrastructure.physics.null_space_erasure``. Orthogonal
projection that removes the marker's spatial-basis component from a
corrupted measurement via a Tikhonov-regularised pseudo-inverse.

Categories:

- 3-D basis (no batch) raises (basis must be 4-D ``[K, C, H, W]``)
- Pure marker input → output ≈ 0 (full erasure of the in-basis signal)
- Anatomy orthogonal to the basis is preserved
- Idempotence: applying the eraser twice gives the same result as once
- Multi-basis (K > 1) extends the erased subspace correctly
- Output shape matches input shape
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.null_space_erasure import NullSpaceMarkerEraser


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_invalid_basis_dim_raises() -> None:
    """Basis must be 4-D ``[K, C, H, W]``."""
    bad = torch.randn(8, 8)  # 2-D
    with pytest.raises(ValueError, match=r"4-D"):
        NullSpaceMarkerEraser(bad)


def test_default_epsilon() -> None:
    """Default Tikhonov ``ε = 1e-6``."""
    basis = torch.randn(1, 1, 8, 8)
    eraser = NullSpaceMarkerEraser(basis)
    assert eraser.epsilon == 1e-6


# ---------------------------------------------------------------------------
# Pure-marker erasure
# ---------------------------------------------------------------------------


def test_pure_marker_input_erased_to_zero() -> None:
    """If input lies fully in the basis, output is approximately zero."""
    torch.manual_seed(0)
    basis = torch.randn(1, 1, 8, 8)
    eraser = NullSpaceMarkerEraser(basis, epsilon=1e-9)
    out = eraser(basis)  # exactly the marker basis
    assert out.abs().max().item() < 1e-3


def test_orthogonal_anatomy_preserved() -> None:
    """Anatomy orthogonal to the basis passes through unchanged.

    Construct a basis vector and an explicitly orthogonal anatomy
    vector; check the eraser passes the anatomy through.
    """
    torch.manual_seed(0)
    # Basis: a single non-zero pixel.
    basis = torch.zeros(1, 1, 4, 4)
    basis[0, 0, 0, 0] = 1.0
    eraser = NullSpaceMarkerEraser(basis, epsilon=1e-9)

    # Anatomy: ones everywhere *except* the basis pixel, which is zero.
    anat = torch.ones(1, 1, 4, 4)
    anat[0, 0, 0, 0] = 0.0
    out = eraser(anat)
    # The anatomy is orthogonal to the basis, so the projection is 0.
    assert torch.allclose(out, anat, atol=1e-5)


def test_mixture_only_marker_component_removed() -> None:
    """``input = anatomy + alpha · basis`` → output ≈ anatomy."""
    torch.manual_seed(0)
    basis = torch.zeros(1, 1, 4, 4)
    basis[0, 0, 0, 0] = 1.0
    eraser = NullSpaceMarkerEraser(basis, epsilon=1e-9)

    anat = torch.ones(1, 1, 4, 4)
    anat[0, 0, 0, 0] = 0.0  # orthogonal to basis
    corrupted = anat + 5.0 * basis
    out = eraser(corrupted)
    assert torch.allclose(out, anat, atol=1e-4)


# ---------------------------------------------------------------------------
# Idempotence
# ---------------------------------------------------------------------------


def test_idempotent() -> None:
    """``eraser(eraser(y)) == eraser(y)``."""
    torch.manual_seed(0)
    basis = torch.randn(2, 1, 4, 4)
    eraser = NullSpaceMarkerEraser(basis, epsilon=1e-9)
    y = torch.randn(3, 1, 4, 4)
    once = eraser(y)
    twice = eraser(once)
    assert torch.allclose(once, twice, atol=1e-4)


# ---------------------------------------------------------------------------
# Multi-basis
# ---------------------------------------------------------------------------


def test_multi_basis_extends_erased_subspace() -> None:
    """K = 2 basis vectors → both erased."""
    basis = torch.zeros(2, 1, 4, 4)
    basis[0, 0, 0, 0] = 1.0
    basis[1, 0, 0, 1] = 1.0
    eraser = NullSpaceMarkerEraser(basis, epsilon=1e-9)
    # Input = sum of both basis vectors.
    y = basis.sum(dim=0, keepdim=True)
    out = eraser(y)
    assert out.abs().max().item() < 1e-3


# ---------------------------------------------------------------------------
# Sanity-shape
# ---------------------------------------------------------------------------


def test_output_shape_matches_input() -> None:
    """Output has the same shape as the corrupted input."""
    basis = torch.randn(1, 1, 8, 8)
    eraser = NullSpaceMarkerEraser(basis)
    y = torch.randn(4, 1, 8, 8)
    out = eraser(y)
    assert out.shape == y.shape


# ---------------------------------------------------------------------------
# Buffers
# ---------------------------------------------------------------------------


def test_basis_registered_as_buffer() -> None:
    """The flattened basis ``M`` is a registered buffer (moves with .to())."""
    basis = torch.randn(2, 1, 4, 4)
    eraser = NullSpaceMarkerEraser(basis)
    names = {n for n, _ in eraser.named_buffers()}
    assert "M" in names


def _complex_basis(device: str = "cpu") -> torch.Tensor:
    m = torch.zeros(1, 1, 16, 16, dtype=torch.complex64, device=device)
    m[0, 0, 4:7, 4:7] = 1.0 + 0.5j
    return m


def test_projection_is_idempotent_for_a_complex_basis() -> None:
    eraser = NullSpaceMarkerEraser(_complex_basis(), epsilon=1e-9)
    once = eraser(torch.randn(3, 1, 16, 16, dtype=torch.complex64) + _complex_basis())
    assert torch.allclose(eraser(once), once, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sync-debug mode needs CUDA")
def test_erasure_does_not_sync_the_host() -> None:
    eraser = NullSpaceMarkerEraser(_complex_basis("cuda"), epsilon=1e-6)
    y = torch.rand(2, 1, 16, 16, dtype=torch.complex64, device="cuda")
    eraser(y)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        eraser(y)
    finally:
        torch.cuda.set_sync_debug_mode("default")
