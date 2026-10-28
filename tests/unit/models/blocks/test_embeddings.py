"""Tests for embedding modules.

Targets ``spectramr.models.blocks.embeddings``:

- ``PatchEmbedding`` — Vision Transformer patch projection
- ``SinusoidalPositionEmbedding`` — diffusion timestep embedding
- ``CoordinatePositionalEncoding`` — multi-frequency Fourier features
- ``FourierKANPositionalEmbedding`` — learnable Fourier positional features

KAN-dependent variants (``AdvancedKANPositionalEmbedding``,
``HybridFourierKANEmbedding``, ``KANTimeEmbedding``) are skipped if the
KAN package is unavailable.
"""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.models.blocks.embeddings import (
    KAN_AVAILABLE,
    CoordinatePositionalEncoding,
    FourierKANPositionalEmbedding,
    PatchEmbedding,
    SinusoidalPositionEmbedding,
)


# ---------------------------------------------------------------------------
# PatchEmbedding
# ---------------------------------------------------------------------------


def test_patch_embedding_output_shape() -> None:
    """``PatchEmbedding`` produces ``(B, num_patches, embed_dim)``."""
    embed = PatchEmbedding(img_size=32, patch_size=8, in_channels=3, embed_dim=64)
    x = torch.randn(2, 3, 32, 32)
    out = embed(x)
    # patches per side = 32/8 = 4 → 16 patches
    assert out.shape == (2, 16, 64)


def test_patch_embedding_num_patches() -> None:
    """``num_patches`` equals ``(img_size / patch_size) ** 2``."""
    embed = PatchEmbedding(img_size=64, patch_size=16)
    assert embed.num_patches == 16
    assert embed.patches_resolution == (4, 4)


def test_patch_embedding_gradient_flows() -> None:
    """Backward through patch embedding reaches input + projection weights."""
    embed = PatchEmbedding(img_size=16, patch_size=4, in_channels=1, embed_dim=8)
    x = torch.randn(1, 1, 16, 16, requires_grad=True)
    embed(x).sum().backward()
    assert x.grad is not None


# ---------------------------------------------------------------------------
# SinusoidalPositionEmbedding
# ---------------------------------------------------------------------------


def test_sinusoidal_embedding_output_shape() -> None:
    """Output shape is ``(B, dim)``."""
    embed = SinusoidalPositionEmbedding(dim=16)
    t = torch.tensor([0.0, 10.0, 100.0])
    out = embed(t)
    assert out.shape == (3, 16)


def test_sinusoidal_embedding_distinct_for_different_timesteps() -> None:
    """Different timesteps produce different embeddings."""
    embed = SinusoidalPositionEmbedding(dim=16)
    out0 = embed(torch.tensor([0.0]))
    out1 = embed(torch.tensor([100.0]))
    assert not torch.allclose(out0, out1)


def test_sinusoidal_embedding_zero_input_has_known_pattern() -> None:
    """At ``t=0``, sin(0)=0 and cos(0)=1 → first half zeros, second half ones."""
    embed = SinusoidalPositionEmbedding(dim=16)
    out = embed(torch.tensor([0.0]))
    # First half is sin(0) = 0
    assert torch.allclose(out[0, :8], torch.zeros(8), atol=1e-6)
    # Second half is cos(0) = 1
    assert torch.allclose(out[0, 8:], torch.ones(8), atol=1e-6)


# ---------------------------------------------------------------------------
# CoordinatePositionalEncoding
# ---------------------------------------------------------------------------


def test_coordinate_encoding_output_dim_with_input() -> None:
    """``out_features = in_features * num_frequencies * 2 + in_features`` when including input."""
    encoding = CoordinatePositionalEncoding(in_features=3, num_frequencies=4, include_input=True)
    assert encoding.out_features == 3 * 4 * 2 + 3
    assert encoding.output_dim(3) == 3 * 4 * 2 + 3


def test_coordinate_encoding_output_dim_no_input() -> None:
    """Without input: ``out_features = in_features * num_frequencies * 2``."""
    encoding = CoordinatePositionalEncoding(in_features=2, num_frequencies=8, include_input=False)
    assert encoding.out_features == 2 * 8 * 2


def test_coordinate_encoding_log_frequencies() -> None:
    """Log-sampled frequencies are powers of 2: ``[1, 2, 4, ..., 2^(L-1)]``."""
    encoding = CoordinatePositionalEncoding(num_frequencies=4, log_sampling=True)
    expected = torch.tensor([1.0, 2.0, 4.0, 8.0])
    assert torch.allclose(encoding.freq_bands, expected)


def test_coordinate_encoding_forward_shape() -> None:
    """Forward output has shape ``[..., out_features]``."""
    encoding = CoordinatePositionalEncoding(in_features=3, num_frequencies=4, include_input=True)
    coords = torch.randn(2, 5, 3)
    out = encoding(coords)
    assert out.shape == (2, 5, encoding.out_features)


def test_coordinate_encoding_zero_coords_yields_known_values() -> None:
    """At ``coords=0``: ``sin(0)=0``, ``cos(0)=1``."""
    encoding = CoordinatePositionalEncoding(in_features=2, num_frequencies=2, include_input=False)
    coords = torch.zeros(1, 2)
    out = encoding(coords)
    # Layout: per-coord-dim, [sin(f1), sin(f2), cos(f1), cos(f2)]
    # All sins are 0, all cos are 1 → output = [0, 0, 1, 1, 0, 0, 1, 1]
    assert (out[0, :].abs() < 1e-6).sum() + (out[0, :].abs() > 0.99).sum() == out.shape[1]


# ---------------------------------------------------------------------------
# FourierKANPositionalEmbedding
# ---------------------------------------------------------------------------


def test_fourier_kan_positional_output_shape() -> None:
    """Output shape is ``(B, seq_len, dim)``."""
    embed = FourierKANPositionalEmbedding(dim=16, num_frequencies=4)
    positions = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    out = embed(positions)
    assert out.shape == (1, 4, 16)


def test_fourier_kan_positional_handles_1d_input() -> None:
    """1-D positions are auto-batched."""
    embed = FourierKANPositionalEmbedding(dim=16, num_frequencies=4)
    positions = torch.tensor([0.0, 1.0, 2.0])
    out = embed(positions)
    assert out.shape == (1, 3, 16)


def test_fourier_kan_positional_gradient_flows() -> None:
    """Backward reaches the learnable frequency parameters."""
    embed = FourierKANPositionalEmbedding(dim=8, num_frequencies=4)
    positions = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    embed(positions).sum().backward()
    assert embed.frequencies.grad is not None
    assert embed.position_scale.grad is not None


# ---------------------------------------------------------------------------
# Sanity-shape matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "img_size, patch_size, embed_dim",
    [(32, 8, 64), (64, 16, 128), (224, 16, 768)],
    ids=["32_8_64", "64_16_128", "224_16_768"],
)
def test_patch_embedding_shape_matrix(img_size: int, patch_size: int, embed_dim: int) -> None:
    """PatchEmbedding handles common ViT geometries."""
    embed = PatchEmbedding(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
    x = torch.randn(1, embed.in_channels, img_size, img_size)
    out = embed(x)
    expected_patches = (img_size // patch_size) ** 2
    assert out.shape == (1, expected_patches, embed_dim)


def test_the_kan_time_embedding_spans_the_horizon_it_is_given():
    """#2442: the default 1000 stood in for every chain length, so the horizon is required."""
    from spectramr.models.blocks.embeddings import KANTimeEmbedding

    with pytest.raises(TypeError, match="max_timesteps"):
        KANTimeEmbedding(16)  # type: ignore[call-arg]
    assert KANTimeEmbedding(16, max_timesteps=29).fourier_kan.max_positions == 29
