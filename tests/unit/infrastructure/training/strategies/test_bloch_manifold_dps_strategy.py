"""Unit tests for BlochManifoldDPSStrategy (Idea 2 / B-2).

The base ``BaseTrainingStrategy.__init__`` stands up heavy infra (AMP,
optimizer stepper, DI services), so we exercise the strategy's *own* logic by
constructing it via ``object.__new__`` and attaching the minimal attributes the
methods read — mirroring the sibling Hamiltonian strategy test. This isolates
the DDPM training step, the config coercion (dict vs typed), and the sampler
composition.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from spectramr.infrastructure.training.strategies.bloch_manifold_dps_strategy import (
    BlochManifoldDPSStrategy,
    _cfg_get,
)
from spectramr.models.diffusion.bloch_manifold_dps_sampler import rescaled_linear_betas


class _DummyScore(nn.Module):
    """Epsilon predictor that tolerates real OR complex [B, C, H, W] inputs.

    DDPM training feeds real-stacked tensors; the DPS sampler feeds complex
    latents (the real generators convert internally). The dummy mirrors that
    by operating on real/imag channels when handed a complex tensor.
    """

    def __init__(self, channels: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.channels = channels

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor | None = None) -> torch.Tensor:
        if torch.is_complex(x):
            r = self.conv(x.real)
            im = self.conv(x.imag)
            return torch.complex(r, im)
        return self.conv(x)


def _bm_block(**overrides) -> dict:
    block = {
        "pulse_sequence": "spgr",
        "param_bounds_T1_ms": (100.0, 5000.0),
        "param_bounds_T2_ms": (10.0, 2000.0),
        "param_bounds_M0": (0.0, 2.0),
        "projection_inner_iterations": 4,
        "projection_gn_damping": 1e-4,
        "projection_every": 1,
        "likelihood_weight_zeta": 0.5,
        "sampling_steps": 50,
        "sampler_type": "ddim",
        "prediction_type": "epsilon",
    }
    block.update(overrides)
    return block


def _make_config(bm_block=None, timesteps=50) -> SimpleNamespace:
    training = SimpleNamespace(
        bloch_manifold_dps=bm_block if bm_block is not None else _bm_block(),
        diffusion=SimpleNamespace(timesteps=timesteps),
    )
    return SimpleNamespace(training=training)


def _build_strategy(config, generator) -> BlochManifoldDPSStrategy:
    """Run the real ``__init__`` with only the heavy base-class init stubbed out."""
    strat = object.__new__(BlochManifoldDPSStrategy)
    strat.config = config
    strat.device = torch.device("cpu")
    strat.env = SimpleNamespace(generator=generator, config=config)
    strat.logging_service = None
    base = BlochManifoldDPSStrategy.__bases__[0]
    with patch.object(base, "__init__", lambda self, *a, **k: None):
        BlochManifoldDPSStrategy.__init__(strat, env=strat.env)
    return strat


@pytest.fixture
def generator():
    torch.manual_seed(0)
    return _DummyScore()


def test_cfg_get_handles_dict_and_object():
    d = {"a": 1}
    assert _cfg_get(d, "a") == 1
    assert _cfg_get(d, "missing", 7) == 7
    ns = SimpleNamespace(a=2)
    assert _cfg_get(ns, "a") == 2
    assert _cfg_get(ns, "missing", 9) == 9


def test_diffusion_buffers_registered(generator):
    strat = _build_strategy(_make_config(timesteps=50), generator)
    assert strat.alphas_cumprod.shape == (50,)
    # Monotone-decreasing cumulative product of alphas.
    assert torch.all(strat.alphas_cumprod[1:] <= strat.alphas_cumprod[:-1] + 1e-6)
    assert torch.isfinite(strat.sqrt_one_minus_alphas_cumprod).all()


def test_q_sample_endpoints(generator):
    strat = _build_strategy(_make_config(timesteps=50), generator)
    x0 = torch.rand(2, 1, 8, 8)
    noise = torch.randn_like(x0)
    t0 = torch.zeros(2, dtype=torch.long)
    xt0 = strat.q_sample(x0, t0, noise)
    # At t=0, x_t ~ x0 (alpha_bar ~ 1).
    assert torch.allclose(xt0, x0, atol=0.2)


def test_to_real_input_complex_and_5d(generator):
    strat = _build_strategy(_make_config(), generator)
    c = torch.randn(2, 1, 8, 8, dtype=torch.complex64)
    out = strat._to_real_input(c)
    assert not torch.is_complex(out)
    assert out.shape[0] == 2 and out.shape[-2:] == (8, 8)
    five = torch.randn(2, 1, 8, 8, 1)
    out5 = strat._to_real_input(five)
    assert out5.shape == (2, 1, 8, 8)


def _attach_data(config, *, dataset_type, coil_processing_mode):
    """Add the minimal data/model/physics blocks needs_ifft_for_visualization reads."""
    config.model = SimpleNamespace(model_type="enhanced_deep_unet", target_domain="image")
    config.data = SimpleNamespace(
        dataset_type=dataset_type,
        normalize_kspace=False,
        output_domain="image",
        coil_processing_mode=coil_processing_mode,
    )
    config.physics = SimpleNamespace(kspace=SimpleNamespace(enable_kspace_recon=False))
    return config


def test_to_real_input_iffts_kspace_target_when_delivered_kspace(generator):
    """A svd-coil k-space-delivering Bloch arm must denoise IMAGES: _to_real_input
    IFFTs the k-space target so x0 is a brain, not a centre-bright k-space blob
    (F-kspace-real, smoke audit 2026-06-13)."""
    from spectramr.infrastructure.physics.fft_ops import ifft2c

    strat = _build_strategy(
        _attach_data(_make_config(), dataset_type="kspace", coil_processing_mode="svd"),
        generator,
    )
    kspace = torch.randn(2, 1, 8, 8, dtype=torch.complex64)
    out = strat._to_real_input(kspace)
    # _to_real_input IFFTs (image domain) then stacks complex -> 2 real channels.
    expected = ifft2c(kspace)
    expected_stacked = torch.cat([expected.real, expected.imag], dim=1)
    assert out.shape == expected_stacked.shape
    assert torch.allclose(out, expected_stacked, atol=1e-5)


def test_to_real_input_is_noop_for_rss_image_arm(generator):
    """exp_p2 uses coil_processing_mode=rss_image → the dataset already emits a
    magnitude image; _to_real_input must NOT re-FFT it (mirror-image regression).
    """
    strat = _build_strategy(
        _attach_data(_make_config(), dataset_type="kspace", coil_processing_mode="rss_image"),
        generator,
    )
    img = torch.rand(2, 1, 8, 8)  # 1-channel magnitude image (rss_image output)
    out = strat._to_real_input(img)
    assert torch.allclose(out, img)  # untouched (no IFFT, no stacking)


def test_compute_losses_ddpm_and_backprop(generator):
    strat = _build_strategy(_make_config(), generator)
    target = torch.rand(2, 1, 8, 8)
    torch.manual_seed(1)
    losses = strat._compute_losses_impl(target, target, epoch=0)
    assert "g_total_loss" in losses and "diffusion_mse" in losses
    total = losses["g_total_loss"]
    assert torch.isfinite(total).all()
    total.backward()
    grads = [p.grad for p in generator.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


class _SpySampler:
    """Records project() calls; pretends to project onto the manifold."""

    def __init__(self) -> None:
        self.project_calls = 0

    def project(self, x0):
        self.project_calls += 1
        return x0 * 0.5  # stand-in for a manifold projection

    def manifold_residual(self, x0):
        return torch.tensor(float(x0.abs().mean()))


def test_validation_exercises_manifold_projection_when_active(generator):
    """exp_p2 regression (pitfall #16): with projection active (projection_every
    small) the validation step must APPLY the Bloch-manifold projection and emit
    val_bloch_manifold_residual, so the primary metric reflects the constrained
    sampler — otherwise this arm and its unconstrained baseline validate
    identically and the projection (the whole novelty) is never exercised."""
    strat = _build_strategy(_make_config(), generator)  # projection_every=1
    spy = _SpySampler()
    strat.build_sampler = lambda: spy
    strat._get_validation_metrics_computer = lambda cfg: SimpleNamespace(
        compute=lambda a, b: {"psnr": 30.0}
    )
    out = strat.validation_step(torch.rand(1, 1, 8, 8), torch.rand(1, 1, 8, 8))
    assert spy.project_calls >= 1, "manifold projection never applied in validation"
    assert "val_bloch_manifold_residual" in out


def test_validation_skips_projection_when_baseline(generator):
    """The unconstrained baseline (projection_every very large) must NOT project
    in validation — so the constrained-vs-unconstrained comparison is real."""
    big = _bm_block(projection_every=10_000)
    strat = _build_strategy(_make_config(bm_block=big), generator)
    spy = _SpySampler()
    strat.build_sampler = lambda: spy
    strat._get_validation_metrics_computer = lambda cfg: SimpleNamespace(
        compute=lambda a, b: {"psnr": 30.0}
    )
    out = strat.validation_step(torch.rand(1, 1, 8, 8), torch.rand(1, 1, 8, 8))
    assert spy.project_calls == 0, "baseline must not project in validation"
    # The raw (unconstrained) manifold residual is still reported for comparison.
    assert "val_bloch_manifold_residual" in out


def test_missing_block_raises(monkeypatch, generator):
    """__init__ must reject a config with no training.bloch_manifold_dps block."""
    cfg = _make_config()
    cfg.training.bloch_manifold_dps = None
    strat = object.__new__(BlochManifoldDPSStrategy)
    # Stub the heavy base __init__ so only the strategy's own guard runs.
    monkeypatch.setattr(
        BlochManifoldDPSStrategy.__bases__[0],
        "__init__",
        lambda self, *a, **k: None,
        raising=False,
    )
    strat.config = cfg
    strat.device = torch.device("cpu")
    strat.env = SimpleNamespace(generator=generator, config=cfg)
    strat.logging_service = None
    with pytest.raises(ValueError, match="bloch_manifold_dps"):
        BlochManifoldDPSStrategy.__init__(strat, env=strat.env)


def test_build_sampler_composes_projector(generator):
    strat = _build_strategy(_make_config(), generator)
    sampler = strat.build_sampler()
    from spectramr.models.diffusion.bloch_manifold_dps_sampler import (
        BlochManifoldDPSSampler,
    )

    assert isinstance(sampler, BlochManifoldDPSSampler)
    assert sampler.zeta == 0.5
    assert sampler.projector.n_inner == 4
    # cached
    assert strat.build_sampler() is sampler


def test_reconstruct_runs_and_no_nan(generator):
    strat = _build_strategy(_make_config(), generator)
    from spectramr.infrastructure.physics.fft_ops import fft2c

    img = torch.complex(torch.rand(1, 1, 8, 8) * 0.05, torch.rand(1, 1, 8, 8) * 0.05)
    y = fft2c(img)
    out = strat.reconstruct(y, num_steps=4)
    assert out.shape == (1, 1, 8, 8)
    assert torch.isfinite(out.real).all() and torch.isfinite(out.imag).all()


def test_dict_config_path(generator):
    """Strategy works when the block is a raw dict (pre-union-mount)."""
    strat = _build_strategy(_make_config(bm_block=_bm_block(likelihood_weight_zeta=1.0)), generator)
    sampler = strat.build_sampler()
    assert sampler.zeta == 1.0


def test_deterministic_losses(generator):
    strat = _build_strategy(_make_config(), generator)
    torch.manual_seed(3)
    target = torch.rand(1, 1, 8, 8)
    torch.manual_seed(99)
    a = strat._compute_losses_impl(target, target, epoch=0)["g_total_loss"]
    torch.manual_seed(99)
    b = strat._compute_losses_impl(target, target, epoch=0)["g_total_loss"]
    assert torch.allclose(a, b)


class _FakeMetricsComputer:
    """Stand-in for the SSOT ValidationMetricsComputer (heavy to build from a
    minimal SimpleNamespace config)."""

    def compute(self, pred, target):  # noqa: ANN001
        assert pred.shape == target.shape
        return {"robust_mri_psnr": torch.tensor(21.0), "ssim": torch.tensor(0.4)}


def test_validation_step_runs_forward_with_timesteps(generator, monkeypatch):
    """F9 (smoke audit 2026-06-04): the inherited base validation called the
    diffusion score net as ``model(x)`` without ``timesteps`` → every val batch
    raised ``forward() missing 1 required positional argument: 'timesteps'``
    (exp_p2, dispatch task 20). The override must run a single ε-prediction
    denoise (model called WITH timesteps) and return ``val_``-prefixed metrics."""
    strat = _build_strategy(_make_config(), generator)
    monkeypatch.setattr(
        strat, "_get_validation_metrics_computer", lambda cfg: _FakeMetricsComputer()
    )
    target = torch.rand(2, 1, 8, 8)
    out = strat.validation_step(target, target)
    assert out, "validation_step returned no metrics"
    assert all(k.startswith("val_") for k in out), out
    assert "val_robust_mri_psnr" in out  # the early-stopping monitor resolves
    assert all(isinstance(v, float) for v in out.values())


def test_validation_step_does_not_call_model_without_timesteps(generator, monkeypatch):
    """Pin the exact regression: the score net must receive ``timesteps``."""
    seen = {}

    class _StrictScore(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 1, 3, padding=1)

        def forward(self, x, timesteps=None):  # noqa: ANN001
            seen["timesteps_is_none"] = timesteps is None
            return self.conv(x)

    strat = _build_strategy(_make_config(), _StrictScore())
    monkeypatch.setattr(
        strat, "_get_validation_metrics_computer", lambda cfg: _FakeMetricsComputer()
    )
    strat.validation_step(torch.rand(2, 1, 8, 8), torch.rand(2, 1, 8, 8))
    assert seen.get("timesteps_is_none") is False, "validation must pass timesteps"


class _ExplodingScore(nn.Module):
    """Declares ``timesteps`` and then fails *inside* the forward."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor | None = None):
        raise TypeError("embedding table dtype mismatch inside the score net")


class _TimestepFreeScore(nn.Module):
    """Genuinely un-conditioned: one positional parameter, no ``timesteps``."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1)
        self.calls: list[int] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls.append(x.shape[0])
        return self.conv(x)


class TestForwardScoreIsIntrospectedNotSwallowed:
    """SAQ-001 (#1189): ``_forward_score`` dispatches on the signature.

    ``try: model(x_t, timesteps=t) except TypeError: model(x_t)`` sat in BOTH
    the training loss and the validation step. It could not tell "no
    ``timesteps`` parameter" from "the forward raised a TypeError deeper down",
    so a broken conditioned path silently degraded to an unconditioned score
    network -- which still trains, and still reports a falling loss.
    """

    def test_a_typeerror_raised_inside_the_forward_propagates(self):
        with pytest.raises(TypeError, match="dtype mismatch inside"):
            BlochManifoldDPSStrategy._forward_score(
                _ExplodingScore(), torch.randn(2, 1, 8, 8), torch.zeros(2, dtype=torch.long)
            )

    def test_a_model_without_the_parameter_is_called_unconditioned(self):
        model = _TimestepFreeScore()
        out = BlochManifoldDPSStrategy._forward_score(
            model, torch.randn(2, 1, 8, 8), torch.zeros(2, dtype=torch.long)
        )
        assert model.calls == [2]
        assert out.shape == (2, 1, 8, 8)

    def test_a_model_with_the_parameter_receives_the_timesteps(self, generator):
        seen: dict = {}
        real_forward = generator.forward

        def _spy(x, timesteps=None):
            seen["timesteps"] = timesteps
            return real_forward(x, timesteps=timesteps)

        generator.forward = _spy
        # Introspect the class's forward, not the patched instance attribute --
        # `_callable_accepts_kwarg` reads whatever `getattr(model, "forward")`
        # returns, and `_spy` declares `timesteps` too, so the dispatch is the
        # same one production takes.
        BlochManifoldDPSStrategy._forward_score(
            generator, torch.randn(2, 1, 8, 8), torch.full((2,), 7, dtype=torch.long)
        )
        assert seen["timesteps"] is not None
        assert int(seen["timesteps"][0]) == 7


def test_the_sampler_walks_the_schedule_the_score_network_trained_on(generator):
    """PLANTED (#2441): the sampler rebuilt its schedule over sampling_steps, not T."""
    strat = _build_strategy(_make_config(_bm_block(sampling_steps=10), timesteps=40), generator)
    sampler = strat.build_sampler()
    assert sampler.num_timesteps == strat.num_timesteps == 40
    assert sampler.sampling_steps == 10
    torch.testing.assert_close(sampler.alphas_cumprod, strat.alphas_cumprod)


def test_a_short_trained_chain_has_a_finite_schedule(generator):
    strat = _build_strategy(_make_config(_bm_block(sampling_steps=10), timesteps=10), generator)
    torch.testing.assert_close(strat.betas, rescaled_linear_betas(10))
    assert torch.isfinite(1.0 / strat.sqrt_alphas_cumprod).all()


def test_more_reverse_steps_than_trained_steps_raises_at_construction(generator):
    with pytest.raises(ValueError, match="sampling_steps=50 must lie in"):
        _build_strategy(_make_config(_bm_block(sampling_steps=50), timesteps=29), generator)
