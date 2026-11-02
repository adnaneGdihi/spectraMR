"""Round-trip checks for the epsilon-prediction schedule math (issue #641).

The owner is ``DiffusionScheduler``. These call the production functions, on CPU
with no generator built, so a wrong index or a mis-keyed alpha shows up as a failure.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from spectramr.infrastructure.training.schedulers.diffusion_scheduler import DiffusionScheduler
from spectramr.infrastructure.training.strategies.diffusion import DiffusionTrainingStrategy

T = 20


@pytest.fixture(params=["linear", "cosine"])
def scheduler(request) -> DiffusionScheduler:
    return DiffusionScheduler(num_timesteps=T, beta_schedule=request.param)


def _pair(sched: DiffusionScheduler, t_vals: list[int]):
    torch.manual_seed(0)
    x0 = torch.randn(len(t_vals), 1, 6, 6)
    eps = torch.randn_like(x0)
    t = torch.tensor(t_vals, dtype=torch.long)
    x_t = sched.q_sample(x0, t, eps)
    return x0, eps, t, x_t


def test_x0_from_true_epsilon_recovers_the_clean_image(scheduler) -> None:
    x0, eps, t, x_t = _pair(scheduler, [0, 1, T // 2, T - 1])
    x0_hat = scheduler.predict_x0_from_eps(x_t, t, eps)
    torch.testing.assert_close(x0_hat, x0, atol=1e-4, rtol=1e-4)


def _double(sched: DiffusionScheduler) -> DiffusionScheduler:
    """float64 schedule: the x0-form reference divides by 1 - abar_t, which cancels in float32."""
    for name in (
        "betas",
        "alphas",
        "alphas_cumprod",
        "sqrt_alphas_cumprod",
        "sqrt_one_minus_alphas_cumprod",
    ):
        setattr(sched, name, getattr(sched, name).double())
    return sched


def test_reverse_mean_from_epsilon_equals_posterior_mean_from_x0(scheduler) -> None:
    """denoise_step's mean (from eps) equals the Ho et al. posterior mean (from x0), per t."""
    sched = _double(scheduler)
    x0, eps, t, x_t = _pair(sched, [0, 3, T // 2, T - 1])
    x0, eps, x_t = x0.double(), eps.double(), x_t.double()
    torch.manual_seed(1)
    out = sched.denoise_step(x_t, eps, t)
    torch.manual_seed(1)
    z = torch.randn_like(x_t)

    abar = sched.alphas_cumprod
    abar_prev = torch.cat([torch.ones(1, dtype=abar.dtype), abar[:-1]])
    # beta is taken as 1 - alpha, the scheduler's own convention: its float32 alphas
    # are not exactly 1 - betas, and at t=0 that gap is a 1e-4 relative error.
    alpha = sched.alphas
    b = (1.0 - alpha[t]).view(-1, 1, 1, 1)
    a = alpha[t].view(-1, 1, 1, 1)
    ab = abar[t].view(-1, 1, 1, 1)
    abp = abar_prev[t].view(-1, 1, 1, 1)
    mu_x0 = (torch.sqrt(abp) * b / (1.0 - ab)) * x0 + (
        torch.sqrt(a) * (1.0 - abp) / (1.0 - ab)
    ) * x_t
    var = (b * (1.0 - abp) / (1.0 - ab)).sqrt()
    nonzero = (t > 0).double().view(-1, 1, 1, 1)
    expected = mu_x0 + nonzero * var * z
    # The schedule tensors are float32-built and then cast, so 1e-5 is the float32 floor;
    # a slipped index or coefficient is 1e-4 or more.
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_ddim_hop_to_the_final_step_returns_x0(scheduler) -> None:
    x0, eps, _, x_t = _pair(scheduler, [T - 1])
    t = T - 1
    out = scheduler.ddim_step(x_t, t, -1, eps)
    torch.testing.assert_close(out, x0, atol=1e-4, rtol=1e-4)


def test_ddim_hop_with_true_epsilon_lands_on_the_forward_marginal(scheduler) -> None:
    x0, eps, _, x_t = _pair(scheduler, [T - 1])
    t, t_prev = T - 1, 7
    out = scheduler.ddim_step(x_t, t, t_prev, eps)
    expected = scheduler.q_sample(x0, torch.tensor([t_prev]), eps)
    torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)


def _oracle_pixel_sampler(sched: DiffusionScheduler, x0: torch.Tensor, steps: int):
    """A host whose network emits the exact epsilon for a known x0 (strided path)."""
    host = MagicMock()
    host.num_timesteps = T
    host.scheduler = sched
    host._resolve_validation_sampling_steps = lambda: steps
    host._maybe_condition_on_input = lambda x, **kw: x

    def forward_chunked(x_in, t_vec, _is_latent, _kw):
        ab = sched.alphas_cumprod[t_vec].view(-1, 1, 1, 1)
        return (x_in - torch.sqrt(ab) * x0) / torch.sqrt(1.0 - ab)

    return host, forward_chunked


def test_pixel_epsilon_sampler_wiring_recovers_x0_with_an_exact_epsilon(scheduler) -> None:
    """The validation loop consumes eps through the scheduler and lands on x0."""
    torch.manual_seed(2)
    x0 = torch.randn(2, 1, 6, 6)
    host, forward_chunked = _oracle_pixel_sampler(scheduler, x0, steps=5)
    out = DiffusionTrainingStrategy._sample_pixel_epsilon(host, x0, {}, x0.shape, forward_chunked)
    torch.testing.assert_close(out, x0, atol=1e-4, rtol=1e-4)


class _RecordingLatentGenerator(torch.nn.Module):
    """Stub latent generator: owns a reverse schedule, records bind and sample calls."""

    def __init__(self) -> None:
        super().__init__()
        self.diffusion = object()
        self.bound: dict | None = None
        self.sample_calls: list[dict] = []

    def set_diffusion_schedule(self, timesteps, beta_schedule, device=None, betas=None) -> None:
        self.bound = {
            "timesteps": timesteps,
            "beta_schedule": beta_schedule,
            "betas": betas.clone(),
        }

    def sample(
        self, shape, device="cpu", contrast_idx=None, condition_image=None, num_inference_steps=None
    ):
        self.sample_calls.append(
            {
                "shape": tuple(shape),
                "contrast_idx": contrast_idx,
                "condition_image": condition_image,
                "num_inference_steps": num_inference_steps,
            }
        )
        return torch.zeros(shape)


def test_predict_binds_the_training_betas_and_samples_the_training_contract() -> None:
    """Predict hands the generator the betas training ran and calls sample() with the batch contract."""
    from spectramr.infrastructure.inference.latent_diffusion_inference_strategy import (
        LatentDiffusionInferenceStrategy,
    )

    diffusion = {"timesteps": T, "noise_schedule": "cosine", "beta_start": 1e-4, "beta_end": 0.02}
    gen = _RecordingLatentGenerator()
    strategy = LatentDiffusionInferenceStrategy(
        gen, torch.device("cpu"), {"training": {"diffusion": diffusion}}
    )
    cond = torch.randn(2, 1, 8, 8)
    contrast = torch.tensor([0, 3])

    out, _ = strategy.run_inference(cond, contrast_idx=contrast)

    train_betas = DiffusionScheduler(
        num_timesteps=T, beta_schedule="cosine", beta_start=1e-4, beta_end=0.02
    ).betas
    assert gen.bound["timesteps"] == T and gen.bound["beta_schedule"] == "cosine"
    assert torch.equal(gen.bound["betas"], train_betas)
    call = gen.sample_calls[0]
    assert call["shape"] == (2, 1, 8, 8)
    assert torch.equal(call["contrast_idx"], contrast)
    assert call["condition_image"] is cond
    assert call["num_inference_steps"] is None
    assert out.shape == (2, 1, 8, 8)


def test_predict_without_training_diffusion_block_raises() -> None:
    from spectramr.domain.exceptions import ConfigurationError
    from spectramr.infrastructure.inference.latent_diffusion_inference_strategy import (
        LatentDiffusionInferenceStrategy,
    )

    strategy = LatentDiffusionInferenceStrategy(
        _RecordingLatentGenerator(), torch.device("cpu"), {}
    )
    with pytest.raises(ConfigurationError, match=r"training\.diffusion"):
        strategy.run_inference(torch.randn(1, 1, 8, 8))
