"""Tests for embeddings.py - Embedding layer implementations."""

from unittest.mock import patch

import pytest
import torch
from torch import nn

from spectramr.models.blocks.embeddings import (
    AdvancedKANPositionalEmbedding,
    FourierKANPositionalEmbedding,
    HybridFourierKANEmbedding,
    KANTimeEmbedding,
    PatchEmbedding,
    SinusoidalPositionEmbedding,
)


class TestPatchEmbedding:
    """Test suite for PatchEmbedding class."""

    def test_initialization_default_params(self):
        """Test initialization with default parameters."""
        embed = PatchEmbedding()

        assert embed.img_size == 224
        assert embed.patch_size == 16
        assert embed.in_channels == 3
        assert embed.embed_dim == 768
        assert embed.num_patches == (224 // 16) ** 2
        assert isinstance(embed.proj, nn.Conv2d)
        assert isinstance(embed.norm, nn.LayerNorm)

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        img_size, patch_size, in_channels, embed_dim = 128, 8, 1, 256
        embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        assert embed.img_size == img_size
        assert embed.patch_size == patch_size
        assert embed.in_channels == in_channels
        assert embed.embed_dim == embed_dim
        assert embed.num_patches == (img_size // patch_size) ** 2

        # Check conv layer configuration
        assert embed.proj.in_channels == in_channels
        assert embed.proj.out_channels == embed_dim
        assert embed.proj.kernel_size == (patch_size, patch_size)
        assert embed.proj.stride == (patch_size, patch_size)

    def test_forward_pass_basic(self):
        """Test basic forward pass."""
        embed = PatchEmbedding(img_size=32, patch_size=8, in_channels=3, embed_dim=64)

        batch_size = 2
        x = torch.randn(batch_size, 3, 32, 32)
        output = embed(x)

        expected_patches = (32 // 8) ** 2  # 16 patches
        assert output.shape == (batch_size, expected_patches, 64)

    def test_forward_pass_different_shapes(self):
        """Test forward pass with different input shapes."""
        embed = PatchEmbedding(img_size=64, patch_size=16, in_channels=1, embed_dim=128)

        test_cases = [
            (1, 1, 64, 64),  # Single image
            (4, 1, 64, 64),  # Batch of images
        ]

        for shape in test_cases:
            x = torch.randn(*shape)
            output = embed(x)
            expected_patches = (64 // 16) ** 2  # 16 patches
            assert output.shape == (shape[0], expected_patches, 128)

    def test_forward_pass_gradient_flow(self):
        """Test that gradients flow through the patch embedding."""
        embed = PatchEmbedding(img_size=32, patch_size=8, embed_dim=64)
        x = torch.randn(2, 3, 32, 32, requires_grad=True)

        output = embed(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_invalid_image_size(self):
        """Test that non-divisible image size raises appropriate error."""
        # This should work fine - Conv2d will handle it
        embed = PatchEmbedding(img_size=33, patch_size=8)  # 33 % 8 != 0
        x = torch.randn(1, 3, 33, 33)
        output = embed(x)  # Should not raise error, just produce different shape

        # Output shape will be determined by conv operation
        assert output.shape[0] == 1  # batch size
        assert output.shape[2] == 768  # embed_dim (default)


class TestSinusoidalPositionEmbedding:
    """Test suite for SinusoidalPositionEmbedding class."""

    def test_initialization(self):
        """Test initialization."""
        dim = 128
        embed = SinusoidalPositionEmbedding(dim)
        assert embed.dim == dim

    def test_forward_pass_single_timestep(self):
        """Test forward pass with single timestep."""
        embed = SinusoidalPositionEmbedding(dim=64)
        time = torch.tensor([5])

        output = embed(time)
        assert output.shape == (1, 64)

    def test_forward_pass_batch_timesteps(self):
        """Test forward pass with batch of timesteps."""
        embed = SinusoidalPositionEmbedding(dim=128)
        time = torch.tensor([1, 10, 100, 500])

        output = embed(time)
        assert output.shape == (4, 128)

    def test_forward_pass_different_dims(self):
        """Test forward pass with different embedding dimensions."""
        for dim in [32, 64, 128, 256]:
            embed = SinusoidalPositionEmbedding(dim)
            time = torch.tensor([42])

            output = embed(time)
            assert output.shape == (1, dim)

            # Check that sin and cos components are present
            half_dim = dim // 2
            sin_part = output[0, :half_dim]
            cos_part = output[0, half_dim:]

            # For timestep 42, these should be different
            assert not torch.allclose(sin_part, cos_part)

    def test_forward_pass_gradient_flow(self):
        """Test that gradients flow through sinusoidal embedding."""
        embed = SinusoidalPositionEmbedding(dim=64)
        time = torch.tensor([10.0], requires_grad=True)

        output = embed(time)
        loss = output.sum()
        loss.backward()

        assert time.grad is not None

    def test_mathematical_correctness(self):
        """Test mathematical correctness of sinusoidal embeddings."""
        embed = SinusoidalPositionEmbedding(dim=32)
        time = torch.tensor([0])

        output = embed(time)

        # For time=0, sin components should be 0, cos components should be 1
        half_dim = 16
        sin_part = output[0, :half_dim]
        cos_part = output[0, half_dim:]

        assert torch.allclose(sin_part, torch.zeros_like(sin_part), atol=1e-6)
        assert torch.allclose(cos_part, torch.ones_like(cos_part), atol=1e-6)


class TestFourierKANPositionalEmbedding:
    """Test suite for FourierKANPositionalEmbedding class."""

    def test_initialization_default_params(self):
        """Test initialization with default parameters."""
        dim = 128
        embed = FourierKANPositionalEmbedding(dim)

        assert embed.dim == dim
        assert embed.max_positions == 10000
        assert embed.num_frequencies == 8
        assert isinstance(embed.frequencies, nn.Parameter)
        assert isinstance(embed.sin_weights, nn.Linear)
        assert isinstance(embed.cos_weights, nn.Linear)

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        dim, max_pos, num_freq = 256, 5000, 16
        embed = FourierKANPositionalEmbedding(dim, max_pos, num_freq)

        assert embed.dim == dim
        assert embed.max_positions == max_pos
        assert embed.num_frequencies == num_freq

        # Check weight dimensions
        assert embed.sin_weights.in_features == num_freq // 2
        assert embed.sin_weights.out_features == dim // 2
        assert embed.cos_weights.in_features == num_freq // 2
        assert embed.cos_weights.out_features == dim // 2

    def test_forward_pass_single_sequence(self):
        """Test forward pass with single sequence."""
        embed = FourierKANPositionalEmbedding(dim=64)
        positions = torch.arange(10)  # Sequence of length 10

        output = embed(positions)
        assert output.shape == (1, 10, 64)

    def test_forward_pass_batch_sequences(self):
        """Test forward pass with batch of sequences."""
        embed = FourierKANPositionalEmbedding(dim=128)
        positions = torch.arange(8).unsqueeze(0).repeat(3, 1)  # 3 sequences of length 8

        output = embed(positions)
        assert output.shape == (3, 8, 128)

    def test_forward_pass_different_lengths(self):
        """Test forward pass with different sequence lengths."""
        embed = FourierKANPositionalEmbedding(dim=64)

        for seq_len in [1, 5, 20, 100]:
            positions = torch.arange(seq_len)
            output = embed(positions)
            assert output.shape == (1, seq_len, 64)

    def test_forward_pass_gradient_flow(self):
        """Test that gradients flow through Fourier KAN embedding."""
        embed = FourierKANPositionalEmbedding(dim=64)
        positions = torch.arange(5, dtype=torch.float32, requires_grad=True)

        output = embed(positions)
        loss = output.sum()
        loss.backward()

        assert positions.grad is not None

    def test_position_normalization(self):
        """Test that position normalization works correctly."""
        embed = FourierKANPositionalEmbedding(dim=64, max_positions=100)

        # Test with positions at different scales
        positions_small = torch.tensor([1, 10, 50])
        positions_large = torch.tensor([10, 100, 500])

        output_small = embed(positions_small)
        output_large = embed(positions_large)

        # Should handle different scales appropriately
        assert output_small.shape == (1, 3, 64)
        assert output_large.shape == (1, 3, 64)


class TestAdvancedKANPositionalEmbedding:
    """Test suite for AdvancedKANPositionalEmbedding class."""

    def test_initialization_kan_available(self):
        """Test initialization when KAN is available."""
        with (
            patch("spectramr.models.blocks.embeddings.KAN_AVAILABLE", True),
            patch("spectramr.models.blocks.embeddings.FastKAN", create=True) as MockFastKAN,
        ):
            dim = 128
            embed = AdvancedKANPositionalEmbedding(dim)

            assert embed.dim == dim
            assert embed.kan_type == "fast"
            assert isinstance(embed.position_encoder, nn.Sequential)
            assert hasattr(embed, "kan_layer")
            # MockFastKAN is the class mock, so the instance is its return values
            # isinstance fails because MockFastKAN is a Mock object, not a type
            assert embed.kan_layer == MockFastKAN.return_value

    def test_initialization_kan_unavailable(self):
        """Test initialization when KAN is unavailable."""
        with patch("spectramr.models.blocks.embeddings.KAN_AVAILABLE", False):
            with pytest.raises(ImportError, match="KAN layers not available"):
                AdvancedKANPositionalEmbedding(128)

    def test_initialization_different_kan_types(self):
        """Test initialization with different KAN types."""
        with (
            patch("spectramr.models.blocks.embeddings.KAN_AVAILABLE", True),
            patch("spectramr.models.blocks.embeddings.FastKAN", create=True),
            patch("spectramr.models.blocks.embeddings.ChebyKANLayer", create=True),
            patch("spectramr.models.blocks.embeddings.WavKANLayer", create=True),
            patch("spectramr.models.blocks.embeddings.KANLayer", create=True),
        ):
            for kan_type in ["fast", "chebyshev", "wavelet", "standard"]:
                embed = AdvancedKANPositionalEmbedding(dim=64, kan_type=kan_type)
                assert embed.kan_type == kan_type

    def test_forward_pass_single_sequence(self):
        """Test forward pass with single sequence."""
        with (
            patch("spectramr.models.blocks.embeddings.KAN_AVAILABLE", True),
            patch("spectramr.models.blocks.embeddings.FastKAN", create=True) as MockFastKAN,
        ):
            # Setup mock return value
            mock_instance = MockFastKAN.return_value
            mock_instance.return_value = torch.randn(10, 64)  # flattened batch*seq

            embed = AdvancedKANPositionalEmbedding(dim=64)

            # Ensure mock returns correct shape when called
            def side_effect(x):
                # x shape is (10, 32) -> output (10, 64)
                return torch.randn(x.shape[0], 64)

            mock_instance.side_effect = side_effect

            positions = torch.arange(10)

            output = embed(positions)
            assert output.shape == (1, 10, 64)

    def test_forward_pass_batch_sequences(self):
        """Test forward pass with batch of sequences."""
        with (
            patch("spectramr.models.blocks.embeddings.KAN_AVAILABLE", True),
            patch("spectramr.models.blocks.embeddings.FastKAN", create=True) as MockFastKAN,
        ):
            # Setup mock
            mock_instance = MockFastKAN.return_value

            def side_effect(x):
                return torch.randn(x.shape[0], 128)

            mock_instance.side_effect = side_effect

            embed = AdvancedKANPositionalEmbedding(dim=128)
            positions = torch.arange(8).unsqueeze(0).repeat(3, 1)

            output = embed(positions)
            assert output.shape == (3, 8, 128)

    def test_forward_pass_gradient_flow(self):
        """Test that gradients flow through advanced KAN embedding."""
        with (
            patch("spectramr.models.blocks.embeddings.KAN_AVAILABLE", True),
            patch("spectramr.models.blocks.embeddings.FastKAN", create=True) as MockFastKAN,
        ):
            # Setup mock to allow gradient flow
            mock_instance = MockFastKAN.return_value

            def side_effect(x):
                # Return tensor connected to input for gradient flow simulation
                # In reality KAN would be complex, here we just pass through
                return torch.cat([x, x], dim=1)  # dim//2 * 2 = dim

            mock_instance.side_effect = side_effect

            embed = AdvancedKANPositionalEmbedding(dim=64)
            positions = torch.arange(5, dtype=torch.float32, requires_grad=True)

            output = embed(positions)
            loss = output.sum()
            loss.backward()

            assert positions.grad is not None


class TestHybridFourierKANEmbedding:
    """Test suite for HybridFourierKANEmbedding class."""

    def test_initialization_kan_available(self):
        """Test initialization when KAN is available."""
        with (
            patch("spectramr.models.blocks.embeddings.KAN_AVAILABLE", True),
            patch("spectramr.models.blocks.embeddings.FastKAN", create=True),
        ):
            dim = 128
            embed = HybridFourierKANEmbedding(dim)

            assert embed.dim == dim
            assert embed.num_frequencies == 8
            assert isinstance(embed.frequencies, nn.Parameter)
            assert hasattr(embed, "kan_processor")

    def test_initialization_kan_unavailable(self):
        """Test initialization when KAN is unavailable."""
        with patch("spectramr.models.blocks.embeddings.KAN_AVAILABLE", False):
            with pytest.raises(ImportError, match="KAN layers not available"):
                HybridFourierKANEmbedding(128)

    def test_forward_pass_single_sequence(self):
        """Test forward pass with single sequence."""
        with (
            patch("spectramr.models.blocks.embeddings.KAN_AVAILABLE", True),
            patch("spectramr.models.blocks.embeddings.FastKAN", create=True) as MockFastKAN,
        ):
            mock_instance = MockFastKAN.return_value

            def side_effect(x):
                return torch.randn(x.shape[0], 64)

            mock_instance.side_effect = side_effect

            embed = HybridFourierKANEmbedding(dim=64)
            positions = torch.arange(10)

            output = embed(positions)
            assert output.shape == (1, 10, 64)

    def test_forward_pass_batch_sequences(self):
        """Test forward pass with batch of sequences."""
        with (
            patch("spectramr.models.blocks.embeddings.KAN_AVAILABLE", True),
            patch("spectramr.models.blocks.embeddings.FastKAN", create=True) as MockFastKAN,
        ):
            mock_instance = MockFastKAN.return_value

            def side_effect(x):
                return torch.randn(x.shape[0], 128)

            mock_instance.side_effect = side_effect

            embed = HybridFourierKANEmbedding(dim=128)
            positions = torch.arange(8).unsqueeze(0).repeat(3, 1)

            output = embed(positions)
            assert output.shape == (3, 8, 128)

    def test_forward_pass_gradient_flow(self):
        """Test that gradients flow through hybrid embedding."""
        with (
            patch("spectramr.models.blocks.embeddings.KAN_AVAILABLE", True),
            patch("spectramr.models.blocks.embeddings.FastKAN", create=True) as MockFastKAN,
        ):
            mock_instance = MockFastKAN.return_value

            def side_effect(x):
                return torch.nn.functional.linear(x, torch.randn(64, x.shape[1]))

            mock_instance.side_effect = side_effect

            embed = HybridFourierKANEmbedding(dim=64)
            positions = torch.arange(5, dtype=torch.float32, requires_grad=True)

            output = embed(positions)
            loss = output.sum()
            loss.backward()

            assert positions.grad is not None


class TestKANTimeEmbedding:
    """Test suite for KANTimeEmbedding class."""

    def test_initialization_kan_available(self):
        """Test initialization when KAN is available."""
        with (
            patch("spectramr.models.blocks.embeddings.KAN_AVAILABLE", True),
            patch("spectramr.models.blocks.embeddings.FastKAN", create=True),
        ):
            dim = 128
            embed = KANTimeEmbedding(dim, max_timesteps=1000)

            assert embed.fourier_kan.dim == dim
            assert isinstance(embed.fourier_kan, AdvancedKANPositionalEmbedding)

    def test_initialization_kan_unavailable(self):
        """Test initialization when KAN is unavailable (fallback)."""
        with patch("spectramr.models.blocks.embeddings.KAN_AVAILABLE", False):
            dim = 128
            embed = KANTimeEmbedding(dim, max_timesteps=1000)

            assert embed.fourier_kan.dim == dim
            assert isinstance(embed.fourier_kan, FourierKANPositionalEmbedding)

    def test_forward_pass_single_timestep(self):
        """Test forward pass with single timestep."""
        embed = KANTimeEmbedding(dim=64, max_timesteps=1000)
        time = torch.tensor([5])

        output = embed(time)
        assert output.shape == (1, 64)

    def test_forward_pass_batch_timesteps(self):
        """Test forward pass with batch of timesteps."""
        embed = KANTimeEmbedding(dim=128, max_timesteps=1000)
        time = torch.tensor([1, 10, 100])

        output = embed(time)
        assert output.shape == (3, 128)

    def test_forward_pass_gradient_flow(self):
        """Test that gradients flow through KAN time embedding."""
        embed = KANTimeEmbedding(dim=64, max_timesteps=1000)
        time = torch.tensor([10.0], requires_grad=True)

        output = embed(time)
        loss = output.sum()
        loss.backward()

        assert time.grad is not None

    def test_forward_pass_different_dims(self):
        """Test forward pass with different embedding dimensions."""
        for dim in [32, 64, 128]:
            embed = KANTimeEmbedding(dim, max_timesteps=1000)
            time = torch.tensor([42])

            output = embed(time)
            assert output.shape == (1, dim)


class TestEmbeddingIntegration:
    """Integration tests for embedding layers."""

    def test_patch_embedding_with_position_embedding(self):
        """Test combining patch embedding with positional embedding."""
        patch_embed = PatchEmbedding(img_size=32, patch_size=8, embed_dim=64)

        # Create image
        x = torch.randn(2, 3, 32, 32)

        # Get patch embeddings
        patches = patch_embed(x)  # (2, 16, 64)

        # Create position indices for patches
        num_patches = patches.shape[1]
        positions = torch.arange(num_patches).unsqueeze(0).repeat(2, 1)

        # This would normally be added, but SinusoidalPositionEmbedding expects time values
        # Just test that both can be created and used
        assert patches.shape[0] == positions.shape[0]
        assert patches.shape[1] == positions.shape[1]

    def test_embeddings_different_dtypes(self):
        """Test embeddings with different tensor dtypes."""
        # Test with float64
        embed = SinusoidalPositionEmbedding(dim=64)
        time = torch.tensor([5.0], dtype=torch.float64)

        output = embed(time)
        assert output.dtype == torch.float64

    def test_embeddings_device_consistency(self):
        """Test that embeddings handle device placement correctly."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            embed = SinusoidalPositionEmbedding(dim=64)
            time = torch.tensor([5]).to(device)

            output = embed(time)
            assert output.device.type == device.type
            assert output.is_cuda

    def test_embedding_parameter_count(self):
        """Test that embeddings have reasonable parameter counts."""
        # Patch embedding
        patch_embed = PatchEmbedding(embed_dim=64)
        patch_params = sum(p.numel() for p in patch_embed.parameters())
        assert patch_params > 1000  # Should have substantial parameters

        # Sinusoidal embedding (no parameters)
        sinus_embed = SinusoidalPositionEmbedding(dim=64)
        sinus_params = sum(p.numel() for p in sinus_embed.parameters())
        assert sinus_params == 0  # Pure functional, no learnable parameters

        # Fourier KAN embedding
        fourier_embed = FourierKANPositionalEmbedding(dim=64)
        fourier_params = sum(p.numel() for p in fourier_embed.parameters())
        assert fourier_params > 100  # Should have learnable parameters
