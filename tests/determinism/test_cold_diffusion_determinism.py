"""C6 determinism contract for the cold-diffusion reverse samplers.

The papers' C6 corollary: the sampler must be a deterministic function of
(measurement, mask, sampler_seed). Three knobs carry the contract —
``sampler_sigma`` (reverse-step noise scale, default 0.0), ``sampler_seed``
(pins the noise stream, reseeded per sample call), ``selection_rule``
(only ``"fixed"`` exists; validated so nothing stochastic slips in).

``PhysicsInformedColdDiffusion.sample`` is covered in all three reverse modes,
and ``ColdDiffusionInferenceStrategy.run_inference``, which reconstructs through
it (#2067), is pinned to add no randomness of its own. Beyond same-in/same-out,
the trust-layer-relevant invariant is asserted at σ>0: noise never touches
the observed support, so ``x_out * obs == measurement * obs`` exactly.
"""

from __future__ import annotations

import logging

import pytest
import torch

from spectramr.infrastructure.inference.cold_diffusion_inference_strategy import (
    ColdDiffusionInferenceStrategy,
)
from spectramr.models.diffusion.kspace_process import PhysicsInformedColdDiffusion

REVERSE_MODES = ("additive", "replace_freeze", "replace_freeze_dc")


class _DeterministicStub(torch.nn.Module):
    """Parameter-free deterministic denoiser: no RNG, no state, no dropout."""

    def forward(self, x, t, **kwargs):
        return 0.9 * x + 0.05


def _diffusion(**kwargs) -> PhysicsInformedColdDiffusion:
    return PhysicsInformedColdDiffusion(
        model=_DeterministicStub(),
        num_timesteps=8,
        max_acceleration=4.0,
        center_fraction=0.08,
        dc_method="hard",
        **kwargs,
        kspace_log_scaled=False,
    )


def _inputs(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    full = torch.randn(1, 2, 32, 32, generator=generator)
    mask = torch.zeros(1, 1, 32, 32)
    mask[..., ::2] = 1.0
    mask[..., 12:20] = 1.0
    return full * mask, mask


class TestSamplerDeterminism:
    @pytest.mark.parametrize("reverse_mode", REVERSE_MODES)
    def test_sigma_zero_is_bitwise_reproducible(self, reverse_mode) -> None:
        """The default contract: two runs (and a fresh instance) agree bit-for-bit."""
        measurement, mask = _inputs()
        diffusion = _diffusion(reverse_mode=reverse_mode)
        out1 = diffusion.sample(measurement, mask)
        out2 = diffusion.sample(measurement, mask)
        out3 = _diffusion(reverse_mode=reverse_mode).sample(measurement, mask)
        assert torch.equal(out1, out2)
        assert torch.equal(out1, out3), "hidden state leaked across instances"

    @pytest.mark.parametrize("reverse_mode", REVERSE_MODES)
    def test_sigma_positive_same_seed_is_bitwise_reproducible(self, reverse_mode) -> None:
        """Per-call reseeding: repeated calls on ONE instance must also agree —
        this is exactly the case a create-once-never-reseed generator fails."""
        measurement, mask = _inputs()
        diffusion = _diffusion(reverse_mode=reverse_mode, sampler_sigma=0.1, sampler_seed=7)
        out1 = diffusion.sample(measurement, mask)
        out2 = diffusion.sample(measurement, mask)
        fresh = _diffusion(
            reverse_mode=reverse_mode, sampler_sigma=0.1, sampler_seed=7
        ).sample(measurement, mask)
        assert torch.equal(out1, out2)
        assert torch.equal(out1, fresh)

    @pytest.mark.parametrize("reverse_mode", REVERSE_MODES)
    def test_sigma_positive_different_seeds_differ(self, reverse_mode) -> None:
        measurement, mask = _inputs()
        out_a = _diffusion(
            reverse_mode=reverse_mode, sampler_sigma=0.1, sampler_seed=7
        ).sample(measurement, mask)
        out_b = _diffusion(
            reverse_mode=reverse_mode, sampler_sigma=0.1, sampler_seed=8
        ).sample(measurement, mask)
        assert not torch.equal(out_a, out_b)

    @pytest.mark.parametrize("reverse_mode", REVERSE_MODES)
    def test_noise_actually_reaches_the_output(self, reverse_mode) -> None:
        """σ>0 must differ from σ=0, else the knob is silently inert."""
        measurement, mask = _inputs()
        clean = _diffusion(reverse_mode=reverse_mode).sample(measurement, mask)
        noisy = _diffusion(
            reverse_mode=reverse_mode, sampler_sigma=0.1, sampler_seed=7
        ).sample(measurement, mask)
        assert not torch.equal(clean, noisy)

    def test_seed_none_is_genuinely_nondeterministic(self) -> None:
        """A fresh torch.Generator has a FIXED default state; seed=None must be
        entropically seeded, not silently reproducible."""
        measurement, mask = _inputs()
        diffusion = _diffusion(reverse_mode="replace_freeze", sampler_sigma=0.1)
        out1 = diffusion.sample(measurement, mask)
        out2 = diffusion.sample(measurement, mask)
        assert not torch.equal(out1, out2)

    @pytest.mark.parametrize("reverse_mode", ("replace_freeze", "replace_freeze_dc"))
    def test_noise_never_touches_the_observed_support(self, reverse_mode) -> None:
        """The trust invariant at σ>0: observed lines stay EXACTLY data-consistent
        (noise is injected after DC, masked off the observed support)."""
        measurement, mask = _inputs()
        out = _diffusion(
            reverse_mode=reverse_mode, sampler_sigma=0.5, sampler_seed=7
        ).sample(measurement, mask)
        obs = mask.expand_as(out)
        assert torch.equal(out * obs, measurement * obs)

    def test_additive_noise_leaves_observed_lines_unchanged(self) -> None:
        """The additive accumulate loop double-adds observed lines wherever the
        process cascade overlaps the measurement mask (the documented additive
        blow-up), so exact pinning is NOT its invariant even at σ=0. What σ>0
        must preserve is weaker: noise never ALTERS the observed lines relative
        to the σ=0 trajectory."""
        measurement, mask = _inputs()
        clean = _diffusion(reverse_mode="additive").sample(measurement, mask)
        noisy = _diffusion(
            reverse_mode="additive", sampler_sigma=0.5, sampler_seed=7
        ).sample(measurement, mask)
        obs = mask.expand_as(clean)
        assert torch.equal(noisy * obs, clean * obs)

    def test_boundedness_survives_sigma_positive(self) -> None:
        """Noise lands BEFORE the magnitude clamp, so the replace_freeze ceiling
        (reverse_clip_ratio × max|observed|) still bounds every coefficient.

        The reference is the TRUE complex modulus on both sides. It used to be
        ``measurement.abs().amax()`` — the elementwise max over interleaved
        Re/Im channels, which under-reads the modulus by up to sqrt(2) and so
        asserted a bound the clamp was never actually enforcing (issue #1281).
        """
        from spectramr.models.diffusion.kspace_process import paired_magnitude

        measurement, mask = _inputs()
        diffusion = _diffusion(
            reverse_mode="replace_freeze", sampler_sigma=5.0, sampler_seed=7
        )
        out = diffusion.sample(measurement, mask)
        ceiling = diffusion.reverse_clip_ratio * paired_magnitude(measurement).amax()
        # The guarantee the radial clamp actually makes: |z| <= ceiling per
        # COEFFICIENT. This is strictly stronger than the elementwise bound
        # asserted below, which now follows from it.
        assert (paired_magnitude(out) <= ceiling + 1e-6).all()
        assert (out.abs() <= ceiling + 1e-6).all()


class _SeededGenerator(torch.nn.Module):
    """A generator whose ``sample()`` is ``PhysicsInformedColdDiffusion.sample``."""

    def __init__(self, sigma: float = 0.0, seed: int | None = None) -> None:
        super().__init__()
        self.sampler_sigma, self.sampler_seed = sigma, seed
        self.num_timesteps = 8
        self._diffusion = _diffusion(
            reverse_mode="replace_freeze", sampler_sigma=sigma, sampler_seed=seed
        )

    def sample(self, measurement, mask=None, inference_timesteps=None, **_kw):
        return self._diffusion.sample(measurement, mask)


def _strategy(sigma: float = 0.0, seed: int | None = None) -> ColdDiffusionInferenceStrategy:
    config = {"training": {"diffusion": {"timesteps": 8, "sampling_steps": 4}}}
    return ColdDiffusionInferenceStrategy(
        _SeededGenerator(sigma, seed), torch.device("cpu"), config
    )


class TestStrategyDeterminism:
    """Predict reconstructs through ``generator.sample()`` (#2067), so its C6
    contract is the sampler's; these pin that the strategy adds no randomness."""

    def test_sigma_zero_is_bitwise_reproducible(self) -> None:
        measurement, mask = _inputs()
        strategy = _strategy()
        out1 = strategy.run_inference(measurement.clone(), mask=mask)
        out2 = strategy.run_inference(measurement.clone(), mask=mask)
        assert torch.equal(out1, out2)

    def test_sigma_positive_same_seed_is_bitwise_reproducible(self) -> None:
        measurement, mask = _inputs()
        strategy = _strategy(0.1, 7)
        out1 = strategy.run_inference(measurement.clone(), mask=mask)
        out2 = strategy.run_inference(measurement.clone(), mask=mask)
        assert torch.equal(out1, out2)
        assert not torch.equal(out1, _strategy().run_inference(measurement.clone(), mask=mask))

    def test_sigma_positive_warns_about_exchangeability_once(self, caplog) -> None:
        """The A9/C7 coupling, read off the generator: warned at construction only."""
        with caplog.at_level(logging.WARNING):
            strategy = _strategy(0.1, 7)
        assert any("recalibrate" in record.message for record in caplog.records)
        caplog.clear()
        measurement, mask = _inputs()
        with caplog.at_level(logging.WARNING):
            strategy.run_inference(measurement.clone(), mask=mask)
        assert not any("recalibrate" in record.message for record in caplog.records)
