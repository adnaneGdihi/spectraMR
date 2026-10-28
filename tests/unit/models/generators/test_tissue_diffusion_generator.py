"""Unit tests for TissueDiffusionGenerator."""

import pytest
import torch

from spectramr.models.generators.tissue_diffusion_generator import TissueDiffusionGenerator


class TestTissueDiffusionGenerator:
    """Tests for tissue-space diffusion generator."""

    @pytest.fixture
    def model(self):
        """Create a small TissueDiffusionGenerator."""
        return TissueDiffusionGenerator(
            in_channels=1,
            out_channels=1,
            tissue_channels=6,
            base_dim=16,
            n_diffusion_steps=10,
        )

    def test_forward_shape(self, model):
        """Forward pass should return (noise_pred, noise_target) with matching shapes."""
        B, C, H, W = 2, 6, 32, 32
        x = torch.randn(B, C, H, W)
        t = torch.randint(0, 10, (B,))

        noise_pred, noise_target = model(x, t=t, tissue_maps=x)
        assert noise_pred.shape == (B, C, H, W)
        assert noise_target.shape == (B, C, H, W)

    def test_q_sample_correct_shape(self, model):
        """Forward diffusion should produce correctly shaped noisy samples."""
        x = torch.randn(2, 6, 16, 16)
        t = torch.tensor([0, 5])
        x_noisy, noise = model.q_sample(x, t)
        assert x_noisy.shape == x.shape
        assert noise.shape == x.shape

    def test_normalize_denormalize_roundtrip(self, model):
        """Normalize → denormalize should approximately recover original values."""
        tissue_dict = {
            "rho": torch.rand(1, 1, 8, 8),
            "t1": torch.rand(1, 1, 8, 8) * 3000 + 500,
            "t2": torch.rand(1, 1, 8, 8) * 200 + 50,
            "t2star": torch.rand(1, 1, 8, 8) * 100 + 20,
            "adc": torch.rand(1, 1, 8, 8) * 0.003 + 0.0001,
            "chi": torch.rand(1, 1, 8, 8) * 2 - 1,
        }
        normalized = model.normalize_tissue(tissue_dict)
        assert normalized.shape == (1, 6, 8, 8)
        assert (normalized >= 0).all() and (normalized <= 1).all()

        recovered = model.denormalize_tissue(normalized)
        for key in tissue_dict:
            assert torch.allclose(
                recovered[key], tissue_dict[key], atol=1e-3
            ), f"Roundtrip failed for {key}"

    def test_no_nans(self, model):
        """Forward pass should not produce NaN values."""
        x = torch.randn(1, 6, 16, 16)
        t = torch.tensor([3])
        noise_pred, noise_target = model(x, t=t, tissue_maps=x)
        assert not torch.isnan(noise_pred).any()

    def test_gradient_flow(self, model):
        """Gradients should flow through the denoiser."""
        x = torch.randn(1, 6, 16, 16, requires_grad=True)
        t = torch.tensor([5])
        noise_pred, noise_target = model(x, t=t, tissue_maps=x)
        loss = noise_pred.sum()
        loss.backward()
        assert x.grad is not None

    def test_model_registration(self):
        """Model should be registered as 'tissue_diffusion_mri'."""
        from spectramr.models.registry import MODEL_REGISTRY

        assert "tissue_diffusion_mri" in MODEL_REGISTRY

    def test_interface_compliance(self, model):
        """Should implement IGenerator interface properties."""
        assert model.name == "TissueDiffusionMRI"
        assert model.in_channels == 1
        assert isinstance(model.get_parameter_count(), int)
        assert model.get_parameter_count() > 0


@pytest.mark.parametrize("n_steps", [10, 1000])
def test_the_denoiser_divides_by_the_generators_own_horizon(n_steps):
    """PLANTED (#2442): the denoiser divided by 1000 while the generator ran n_steps."""
    model = TissueDiffusionGenerator(tissue_channels=6, base_dim=16, n_diffusion_steps=n_steps)
    seen = []
    model.denoiser.time_embed.register_forward_pre_hook(lambda _m, args: seen.append(args[0]))
    x, t = torch.randn(2, 6, 16, 16), torch.tensor([0, n_steps - 1])
    model(x, t=t, tissue_maps=x)
    assert torch.equal(seen[0], t.float() / n_steps)
