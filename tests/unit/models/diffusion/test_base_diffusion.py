import pytest
import torch
import torch.nn.functional as F
from torch import nn

from spectramr.models.diffusion.base_diffusion import (
    Diffusion,
    DoubleConvWithTime,
    DownWithTime,
    UpWithTime,
    cosine_beta_schedule,
    linear_beta_schedule,
)


class MockModel(nn.Module):
    def __init__(self, out_channels=1):
        super().__init__()
        self.out_channels = out_channels
        self.linear = nn.Linear(1, 1)  # Dummy parameter to have a device

    def forward(self, x, t, cond=None):
        return torch.randn_like(x)


class TestBaseDiffusion:
    def test_beta_schedules(self):
        timesteps = 100
        linear_betas = linear_beta_schedule(timesteps)
        assert len(linear_betas) == timesteps
        assert linear_betas[0] < linear_betas[-1]

        cosine_betas = cosine_beta_schedule(timesteps)
        assert len(cosine_betas) == timesteps
        assert torch.all(cosine_betas >= 0) and torch.all(cosine_betas <= 1)

    def test_diffusion_init(self):
        diffusion = Diffusion(timesteps=100, beta_schedule="linear")
        assert diffusion.timesteps == 100
        assert len(diffusion.betas) == 100

        diffusion_cosine = Diffusion(timesteps=100, beta_schedule="cosine")
        assert len(diffusion_cosine.betas) == 100

    def test_q_sample(self):
        diffusion = Diffusion(timesteps=100)
        x_start = torch.randn(2, 1, 32, 32)
        t = torch.randint(0, 100, (2,))

        x_noisy = diffusion.q_sample(x_start, t)
        assert x_noisy.shape == x_start.shape

        # Test with fixed noise
        noise = torch.randn_like(x_start)
        x_noisy_fixed = diffusion.q_sample(x_start, t, noise=noise)
        assert x_noisy_fixed.shape == x_start.shape

    def test_p_losses(self):
        diffusion = Diffusion(timesteps=100)
        model = MockModel()
        x_start = torch.randn(2, 1, 32, 32)
        t = torch.randint(0, 100, (2,))

        loss = diffusion.p_losses(model, x_start, t, loss_type="l1")
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar loss

        loss_l2 = diffusion.p_losses(model, x_start, t, loss_type="l2")
        assert isinstance(loss_l2, torch.Tensor)

    def test_p_sample(self):
        diffusion = Diffusion(timesteps=100)
        model = MockModel()
        x = torch.randn(2, 1, 32, 32)
        t = torch.tensor([99, 99])

        out = diffusion.p_sample(model, x, t, t_index=99)
        assert out.shape == x.shape

    def test_p_sample_step_drops_noise_only_at_timestep_zero(self):
        """``t_index`` is the timestep: t=0 returns the mean, any later step samples (#482)."""
        diffusion = Diffusion(timesteps=100)
        x = torch.randn(2, 1, 8, 8)
        noise = torch.randn_like(x)
        final = [diffusion.p_sample_step(x, torch.tensor([0, 0]), noise, 0) for _ in range(2)]
        assert torch.equal(final[0], final[1])
        first = [diffusion.p_sample_step(x, torch.tensor([99, 99]), noise, 99) for _ in range(2)]
        assert not torch.equal(first[0], first[1])

    def test_p_sample_rejects_model_without_t_arg(self):
        """F-DIFFUSION-T / 2026-05-20 — diffusion p_sample must reject
        models whose forward signature can't accept the timestep
        argument, with a TypeError that names the model class and the
        YAML knob the user needs to flip.

        Smoke run 20260519 surfaced ``UNet.forward() takes 2 positional
        arguments but 3 were given`` from ``laplace_diffusion_generator``
        and ``rician_diffusion`` whose YAMLs wired up a plain
        ``standard_unet`` (no time conditioning). Without the explicit
        guard, the validation generator failed silently and the
        cascading validator returned ``None`` for every R-level.
        """
        import pytest

        class _PlainUNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 1)

            def forward(self, x):  # NO ``t`` arg
                return self.conv(x)

        diffusion = Diffusion(timesteps=10)
        x = torch.randn(1, 1, 8, 8)
        t = torch.tensor([0])

        with pytest.raises(
            TypeError, match=r"does not accept a timestep|diffusion_unet"
        ):
            diffusion.p_sample(_PlainUNet(), x, t, t_index=0)

    def test_p_sample_loop(self):
        diffusion = Diffusion(timesteps=10)  # Small timesteps for speed
        model = MockModel()
        shape = (2, 1, 16, 16)

        out = diffusion.p_sample_loop(model, shape)
        assert out.shape == shape

    def test_blocks(self):
        # DoubleConvWithTime
        block = DoubleConvWithTime(in_channels=16, out_channels=32, time_emb_dim=10)
        x = torch.randn(2, 16, 32, 32)
        t = torch.randn(2, 10)
        out = block(x, t)
        assert out.shape == (2, 32, 32, 32)

        # DownWithTime
        down = DownWithTime(in_channels=16, out_channels=32, time_emb_dim=10)
        out_down = down(x, t)
        assert out_down.shape == (2, 32, 16, 16)

        # UpWithTime
        up = UpWithTime(in_channels=32, out_channels=16, time_emb_dim=10)
        x1 = torch.randn(2, 32, 16, 16)  # From down path (smaller)
        x2 = torch.randn(2, 16, 32, 32)  # From skip connection (larger)
        # UpWithTime takes (x1, x2, t). x1 is upsampled to match x2.
        # x1 input channels is 32. Up reduces to 16.
        # Concatenates with x2 (16 channels) -> 32 channels.
        # DoubleConv reduces to out_channels (16).

        out_up = up(x1, x2, t)
        assert out_up.shape == (2, 16, 32, 32)


class _HalfInput(nn.Module):
    def forward(self, x, t, cond=None):
        return 0.5 * x


class TestPLossesRegistryRouting:
    """WS-6 plan #6: p_losses elementary dispatch is registry-routed.

    The rewiring must be numerically identical to the replaced
    ``torch.nn.functional`` calls and must keep raising (never defaulting)
    on unknown ``loss_type`` values.
    """

    def _setup(self):
        diffusion = Diffusion(timesteps=10, device="cpu")
        model = _HalfInput()
        torch.manual_seed(0)
        x_start = torch.randn(2, 1, 8, 8)
        noise = torch.randn_like(x_start)
        t = torch.tensor([3, 7])
        return diffusion, model, x_start, noise, t

    def test_matches_replaced_functionals(self):
        diffusion, model, x_start, noise, t = self._setup()
        x_noisy = diffusion.q_sample(x_start, t, noise=noise)
        predicted = model(x_noisy, t)
        expected = {
            "l1": F.l1_loss(predicted, noise),
            "l2": F.mse_loss(predicted, noise),
            "huber": F.smooth_l1_loss(predicted, noise),
        }
        for name, exp in expected.items():
            got = diffusion.p_losses(model, x_start, t, noise=noise, loss_type=name)
            assert torch.allclose(got, exp), name

    def test_unknown_loss_type_raises(self):
        diffusion, model, x_start, noise, t = self._setup()
        with pytest.raises(ValueError, match="Unsupported loss type"):
            diffusion.p_losses(model, x_start, t, noise=noise, loss_type="nope")


class TestDDIMStep:
    """``Diffusion.ddim_step`` — the skip-capable reverse step.

    ``p_sample_step`` is a strict DDPM ancestral step (it reads ``betas_t`` /
    ``sqrt_recip_alphas_t``), so it is only valid for ``t -> t-1``. Strided
    validation sampling (``validation.sampler_steps``) needs a step that is
    exact for an arbitrary ``t -> t_prev`` jump.
    """

    def test_ddim_step_is_exact_inverse_of_q_sample(self):
        # With eta=0 and the TRUE noise, DDIM's x0 estimate is exact, so one
        # step from t to t_prev must reproduce q_sample(x0, t_prev, eps)
        # regardless of how many timesteps were skipped.
        diff = Diffusion(timesteps=100, beta_schedule="linear", device="cpu")
        torch.manual_seed(0)
        x0 = torch.randn(2, 3, 8, 8)
        eps = torch.randn_like(x0)
        t = torch.full((2,), 90, dtype=torch.long)
        t_prev = torch.full((2,), 60, dtype=torch.long)  # a 30-step skip

        x_t = diff.q_sample(x0, t, eps)
        out = diff.ddim_step(x_t, t, t_prev, eps)
        expected = diff.q_sample(x0, t_prev, eps)

        assert torch.allclose(out, expected, atol=1e-4)

    def test_ddim_step_final_hop_returns_x0(self):
        # t_prev < 0 means "the last hop": alpha_bar_prev = 1, so the result is
        # the clean x0 estimate with no residual noise term.
        diff = Diffusion(timesteps=100, beta_schedule="linear", device="cpu")
        torch.manual_seed(0)
        x0 = torch.randn(2, 3, 8, 8)
        eps = torch.randn_like(x0)
        t = torch.full((2,), 40, dtype=torch.long)
        x_t = diff.q_sample(x0, t, eps)

        out = diff.ddim_step(x_t, t, torch.full((2,), -1, dtype=torch.long), eps)

        assert torch.allclose(out, x0, atol=1e-4)

    def test_ddim_step_is_deterministic_at_eta_zero(self):
        diff = Diffusion(timesteps=50, beta_schedule="cosine", device="cpu")
        torch.manual_seed(0)
        x_t = torch.randn(1, 2, 4, 4)
        eps = torch.randn_like(x_t)
        t = torch.full((1,), 30, dtype=torch.long)
        t_prev = torch.full((1,), 20, dtype=torch.long)

        a = diff.ddim_step(x_t, t, t_prev, eps)
        b = diff.ddim_step(x_t, t, t_prev, eps)

        assert torch.equal(a, b)

    def test_eta_above_zero_injects_noise_and_scales_with_eta(self):
        # eta is new API surface with no in-repo caller yet; exercise it so the
        # stochastic branch is not advertised-but-untested (pitfall #16).
        diff = Diffusion(timesteps=100, beta_schedule="linear", device="cpu")
        torch.manual_seed(0)
        x0 = torch.randn(4, 1, 16, 16)
        eps = torch.randn_like(x0)
        t = torch.full((4,), 80, dtype=torch.long)
        t_prev = torch.full((4,), 60, dtype=torch.long)
        x_t = diff.q_sample(x0, t, eps)

        torch.manual_seed(1)
        a = diff.ddim_step(x_t, t, t_prev, eps, eta=1.0)
        torch.manual_seed(2)
        b = diff.ddim_step(x_t, t, t_prev, eps, eta=1.0)
        det = diff.ddim_step(x_t, t, t_prev, eps, eta=0.0)

        # Stochastic: two draws differ, and both differ from the eta=0 path.
        assert not torch.equal(a, b)
        assert not torch.allclose(a, det, atol=1e-5)
        # Larger eta => more spread around the deterministic trajectory.
        torch.manual_seed(3)
        small = diff.ddim_step(x_t, t, t_prev, eps, eta=0.1)
        torch.manual_seed(3)
        large = diff.ddim_step(x_t, t, t_prev, eps, eta=1.0)
        assert (large - det).abs().mean() > (small - det).abs().mean()

    def test_final_hop_is_deterministic_even_when_eta_is_set(self):
        # alpha_bar_prev = 1 on the last hop => sigma = 0, so the x_0 estimate
        # must not be perturbed regardless of eta.
        diff = Diffusion(timesteps=100, beta_schedule="linear", device="cpu")
        torch.manual_seed(0)
        x0 = torch.randn(2, 1, 8, 8)
        eps = torch.randn_like(x0)
        t = torch.full((2,), 50, dtype=torch.long)
        x_t = diff.q_sample(x0, t, eps)
        final = torch.full((2,), -1, dtype=torch.long)

        out = diff.ddim_step(x_t, t, final, eps, eta=1.0)

        assert torch.allclose(out, x0, atol=1e-4)


# --------------------------------------------------------------------------
# A schedule NAME is not a schedule (2026-07 ldm cohort review).
#
# DiffusionScheduler (forward/training) builds `cosine` as cos(pi/2 * t/T)^2
# with s=0 and clamp [0, 0.999]; base_diffusion.cosine_beta_schedule builds the
# Nichol-Dhariwal s=0.008 variant with clip [1e-4, 0.9999]. Binding the reverse
# process to the forward one BY NAME therefore left sampling inverting a
# trajectory training never ran. Diffusion now accepts explicit betas.
# --------------------------------------------------------------------------


def test_the_two_cosine_implementations_really_do_differ() -> None:
    """Pins the premise -- if these ever converge, the betas hand-over is moot."""
    import torch

    from spectramr.infrastructure.training.schedulers.diffusion_scheduler import (
        DiffusionScheduler,
    )
    from spectramr.models.diffusion.base_diffusion import cosine_beta_schedule

    fwd = DiffusionScheduler(num_timesteps=1000, beta_schedule="cosine").betas
    rev = cosine_beta_schedule(1000)
    assert not torch.allclose(fwd, rev, atol=1e-6)


def test_explicit_betas_are_used_verbatim() -> None:
    import torch

    from spectramr.models.diffusion.base_diffusion import Diffusion

    betas = torch.linspace(0.01, 0.3, 50)
    d = Diffusion(timesteps=50, beta_schedule="cosine", betas=betas)
    torch.testing.assert_close(d.betas.cpu(), betas)
    # Derived quantities follow the supplied betas, not the named schedule.
    torch.testing.assert_close(d.alphas.cpu(), 1.0 - betas)


def test_forward_and_reverse_schedules_agree_after_handover() -> None:
    """The whole point: one schedule, owned by the forward process."""
    import torch

    from spectramr.infrastructure.training.schedulers.diffusion_scheduler import (
        DiffusionScheduler,
    )
    from spectramr.models.diffusion.base_diffusion import Diffusion

    fwd = DiffusionScheduler(num_timesteps=200, beta_schedule="cosine")
    rev = Diffusion(timesteps=200, beta_schedule="cosine", betas=fwd.betas)

    torch.testing.assert_close(rev.betas.cpu(), fwd.betas.cpu())
    torch.testing.assert_close(rev.alphas_cumprod.cpu(), fwd.alphas_cumprod.cpu())


def test_named_schedule_still_works_when_no_betas_supplied() -> None:
    """Every non-LDM caller keeps its historical schedule untouched."""
    import torch

    from spectramr.models.diffusion.base_diffusion import Diffusion, cosine_beta_schedule

    d = Diffusion(timesteps=100, beta_schedule="cosine")
    torch.testing.assert_close(d.betas.cpu(), cosine_beta_schedule(100))


def test_mismatched_betas_length_raises() -> None:
    import pytest
    import torch

    from spectramr.models.diffusion.base_diffusion import Diffusion

    with pytest.raises(ValueError, match="length timesteps"):
        Diffusion(timesteps=100, beta_schedule="cosine", betas=torch.linspace(0.1, 0.2, 7))
