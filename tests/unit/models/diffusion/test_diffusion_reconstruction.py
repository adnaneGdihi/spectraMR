from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from spectramr.models.diffusion.diffusion_reconstruction import (
    DataConsistency,
    DiffusionPosteriorSampling,
    DiffusionPrior,
    PnPReconstruction,
    ReconstructionWithDiffusionPrior,
    REDReconstruction,
    create_diffusion_reconstruction_config,
)


class MockModule(nn.Module):
    def __init__(self, side_effect=None):
        super().__init__()
        self.mock = MagicMock()
        if side_effect:
            self.mock.side_effect = side_effect
        self.device = torch.device("cpu")

    def forward(self, *args, **kwargs):
        return self.mock(*args, **kwargs)


class MockScheduler:
    def alpha_t(self, t):
        # Return tensor of shape t.shape
        # alpha decays from 1 to 0
        return torch.ones_like(t, dtype=torch.float32) * 0.5

    def alpha_t_prev(self, t):
        return torch.ones_like(t, dtype=torch.float32) * 0.6

    def sigma_t(self, t):
        return torch.ones_like(t, dtype=torch.float32) * 0.1


from spectramr.infrastructure.physics.fft_ops import fft2c


def mock_forward_operator(x):
    # Use fft2c to match ifft2c in DataConsistency
    return fft2c(x)


class TestDiffusionReconstruction:
    @pytest.fixture
    def mock_diffusion_model(self):
        def side_effect(x, t, cond=None):
            # Return noise prediction of same shape as x
            # Must support grad for PosteriorSampling
            return torch.zeros_like(x)

        return MockModule(side_effect=side_effect)

    @pytest.fixture
    def mock_scheduler(self):
        return MockScheduler()

    def test_diffusion_prior(self, mock_diffusion_model, mock_scheduler):
        prior = DiffusionPrior(
            mock_diffusion_model, mock_scheduler, num_inference_steps=5
        )
        x_t = torch.randn(2, 1, 32, 32)
        t = torch.tensor([4, 4])

        # Test denoise_step
        out = prior.denoise_step(x_t, t)
        assert out.shape == x_t.shape

        # Test sample_prior
        sample = prior.sample_prior((2, 1, 32, 32))
        assert sample.shape == (2, 1, 32, 32)

    def test_data_consistency(self):
        dc = DataConsistency(mock_forward_operator, lambda_dc=0.5)
        x = torch.randn(2, 1, 32, 32)
        y = mock_forward_operator(x)  # Perfect consistency
        mask = torch.ones(2, 1, 32, 32)

        out = dc(x, y, mask)
        # Should be close to x since y matches x
        assert out.shape == x.shape
        assert torch.allclose(out, x, atol=1e-5)

    def test_pnp_reconstruction(self, mock_diffusion_model, mock_scheduler):
        prior = DiffusionPrior(mock_diffusion_model, mock_scheduler)
        dc = DataConsistency(mock_forward_operator)
        pnp = PnPReconstruction(prior, dc, num_iterations=2)

        y = torch.randn(2, 1, 32, 32, dtype=torch.complex64)
        mask = torch.ones(2, 1, 32, 32)

        out = pnp.reconstruct(y, mask)
        assert out.shape == (2, 1, 32, 32)

    def test_red_reconstruction(self):
        denoiser = MockModule(side_effect=lambda x: x)  # Identity denoiser
        dc = DataConsistency(mock_forward_operator)
        red = REDReconstruction(denoiser, dc, num_iterations=2)

        y = torch.randn(2, 1, 32, 32, dtype=torch.complex64)
        mask = torch.ones(2, 1, 32, 32)

        out = red.reconstruct(y, mask)
        assert out.shape == (2, 1, 32, 32)

    def test_diffusion_posterior_sampling(self, mock_diffusion_model, mock_scheduler):
        # Posterior sampling requires gradients
        # mock_diffusion_model returns zeros_like(x), which has grad_fn if x has grad

        # We need to ensure mock_diffusion_model output is connected to input x in computation graph
        # if we want meaningful gradients. But for shape testing, just returning a tensor is enough
        # provided autograd doesn't crash on None gradients.
        # But autograd.grad will fail if output doesn't depend on input.

        def grad_side_effect(x, t, cond=None):
            # Make output depend on x
            return x * 0.1

        mock_diffusion_model.mock.side_effect = grad_side_effect

        dps = DiffusionPosteriorSampling(
            mock_diffusion_model,
            mock_scheduler,
            mock_forward_operator,
            num_posterior_steps=2,
        )

        y = torch.randn(2, 1, 32, 32, dtype=torch.complex64)
        mask = torch.ones(2, 1, 32, 32)

        out = dps.reconstruct(y, mask)
        assert out.shape == (2, 1, 32, 32)

    def test_reconstruction_wrapper(self, mock_diffusion_model, mock_scheduler):
        # Test PnP wrapper
        recon = ReconstructionWithDiffusionPrior(
            method="pnp",
            diffusion_model=mock_diffusion_model,
            noise_scheduler=mock_scheduler,
            forward_operator=mock_forward_operator,
            num_iterations=1,
        )
        y = torch.randn(2, 1, 32, 32, dtype=torch.complex64)
        out = recon(y)
        assert out.shape == (2, 1, 32, 32)

    def test_config_creation(self):
        config = create_diffusion_reconstruction_config(method="pnp")
        assert config["method"] == "pnp"
        assert "num_iterations" in config


# --- #801: the high-level interface is an nn.Module -------------------------


def test_reconstruction_with_diffusion_prior_is_an_nn_module():
    """It was a plain class holding ``self.reconstructor``, registered as a model."""
    assert issubclass(ReconstructionWithDiffusionPrior, nn.Module)


def test_call_is_not_shadowed_and_forward_exists():
    """``__call__`` WAS the entry point here; it is now ``forward``.

    Left as ``__call__`` on an ``nn.Module`` it would shadow ``_call_impl``, so
    hooks would never fire on the one method the class actually implements.
    """
    assert "__call__" not in vars(ReconstructionWithDiffusionPrior)
    assert ReconstructionWithDiffusionPrior.__call__ is nn.Module.__call__
    assert callable(ReconstructionWithDiffusionPrior.forward)


@pytest.mark.parametrize("method", ["pnp", "red", "posterior"])
def test_the_wrapped_diffusion_models_weights_reach_parameters(method):
    """The registration is load-bearing, not cosmetic.

    Probed with a real two-layer model rather than the ``None`` default, which
    would report zero parameters whether or not the child registered.
    """
    dm = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.Conv2d(4, 1, 3, padding=1))
    expected = sum(1 for _ in dm.parameters())
    model = ReconstructionWithDiffusionPrior(method=method, diffusion_model=dm)

    assert "reconstructor" in dict(model.named_children())
    assert sum(1 for _ in model.parameters()) == expected
    assert all(k.startswith("reconstructor.") for k in model.state_dict())
