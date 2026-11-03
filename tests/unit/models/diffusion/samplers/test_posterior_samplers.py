"""Tests for DPS / ΠGDM / DDS samplers (PR-20)."""

from __future__ import annotations

import torch

from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
from spectramr.models.diffusion.samplers import get_sampler, list_available
from spectramr.models.diffusion.samplers.posterior_samplers import (
    DDSReconSampler,
    DDSSampler,
    DPSSampler,
    DynamicDPSSampler,
    PiGDMSampler,
)


def test_samplers_registered() -> None:
    available = list_available()
    for name in ("dps_posterior", "pi_gdm", "dds", "dynamic_dps"):
        assert name in available, f"{name} missing from sampler registry"


# ---------------------------------------------------------------------------
# DDS un-trap: World-A adapter reachable via the generator dispatch convention
# (get_sampler(model=..., num_timesteps=..., ...) + sample(measurement, mask)).
# Regression: the bare DDSSampler crashed on `model=` and the mismatched
# sample() signature; the adapter fixes the dead-knob-that-crashes (pitfall #15).
# ---------------------------------------------------------------------------


class _ScoreModelStub(torch.nn.Module):
    """Minimal model satisfying the DDS contract: ``_model_forward(x_t, t) -> eps``
    plus ``sqrt_alphas_cumprod`` buffers (mirrors ScoreBasedDiffusionGenerator)."""

    def __init__(self, n: int = 50) -> None:
        super().__init__()
        self.timesteps = n
        betas = torch.linspace(1e-4, 0.02, n)
        ac = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("sqrt_alphas_cumprod", ac.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - ac).sqrt())

    def _model_forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Deterministic zero-noise prediction → reverse loop is driven by the
        # CG data-consistency step alone, making the mechanism testable.
        return torch.zeros_like(x_t)


class _NotAScoreModel(torch.nn.Module):
    """Lacks the DDS contract (no _model_forward / no alpha buffers)."""


def test_dds_adapter_construction_tolerates_generator_kwargs() -> None:
    """The exact kwargs kspace_cold_diffusion_generator:2848 passes must NOT crash.

    Pre-fix, ``DDSSampler(model=..., num_timesteps=..., max_acceleration=..., ...)``
    raised ``TypeError: unexpected keyword argument 'model'``.
    """
    DDSReconSampler(
        model=_ScoreModelStub(),
        num_timesteps=50,
        max_acceleration=8.0,
        center_fraction=0.08,
        dc_method="hard",
        dc_weight=1.0,
        sampling_steps=5,
    )


def test_dds_adapter_resolves_via_get_sampler_model_convention() -> None:
    """get_sampler('dds', model=..., <generator kwargs>) returns the adapter."""
    sampler = get_sampler(
        "dds",
        model=_ScoreModelStub(),
        num_timesteps=50,
        max_acceleration=8.0,
        center_fraction=0.08,
        dc_method="hard",
        dc_weight=1.0,
        sampling_steps=5,
    )
    assert sampler.__class__.__name__ == "DDSReconSampler"


def test_dds_adapter_sample_improves_measurement_consistency() -> None:
    """sample(measurement, mask) runs and the CG-DC pulls the estimate toward the
    measurement — the mechanism-fires probe (output more consistent than x_T)."""
    torch.manual_seed(0)
    shape = (1, 1, 8, 8)
    img = torch.randn(*shape)
    full_k = fft2c(img)
    mask = torch.zeros(1, 1, 8, 8)
    mask[..., ::2, :] = 1.0  # undersample
    measurement = full_k * mask

    sampler = get_sampler(
        "dds",
        model=_ScoreModelStub(),
        num_timesteps=50,
        max_acceleration=4.0,
        center_fraction=0.08,
        sampling_steps=8,
        cg_iters=5,
    )
    out = sampler.sample(measurement, mask)
    assert out.shape == shape
    assert torch.isfinite(out).all()
    # Measurement consistency of the reconstruction vs the zero image baseline.
    resid_out = ((fft2c(out) - measurement) * mask).abs().sum()
    resid_zero = (measurement * mask).abs().sum()  # ||A·0 - y|| on sampled lines
    assert resid_out < resid_zero


def test_dds_adapter_rejects_non_score_model() -> None:
    """DDS requires a score/eps model; a model lacking the contract fails loud."""
    import pytest

    sampler = DDSReconSampler(model=_NotAScoreModel(), num_timesteps=50, sampling_steps=3)
    mask = torch.ones(1, 1, 8, 8)
    measurement = torch.complex(torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8))
    with pytest.raises((AttributeError, TypeError, ValueError)):
        sampler.sample(measurement, mask)


# ---------------------------------------------------------------------------
# DynamicDPS: top-3 dynamic-guidance variants (adaptive_step / pigdm / ncchi),
# one validated mode-dispatched World-A sampler.
# ---------------------------------------------------------------------------
import pytest  # noqa: E402

_DDPS_MODES = ["adaptive_step", "pigdm", "ncchi"]


def _undersampled_measurement(seed: int = 0):
    torch.manual_seed(seed)
    shape = (1, 1, 8, 8)
    measurement = fft2c(torch.randn(*shape))
    mask = torch.zeros(1, 1, 8, 8)
    mask[..., ::2, :] = 1.0
    return measurement * mask, mask, shape


def _ddps(mode: str, **kw):
    if mode == "ncchi":
        kw.setdefault("sigma", 0.01)
        kw.setdefault("n_coils", 4)
    return get_sampler(
        "dynamic_dps", model=_ScoreModelStub(), num_timesteps=50, sampling_steps=8,
        guidance_mode=mode, guidance_scale=0.5, **kw,
    )


def test_dynamic_dps_resolves_via_get_sampler() -> None:
    assert isinstance(_ddps("adaptive_step"), DynamicDPSSampler)


def test_dynamic_dps_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="guidance_mode"):
        DynamicDPSSampler(model=_ScoreModelStub(), guidance_mode="telepathy")


def test_dynamic_dps_ncchi_requires_sigma() -> None:
    """The novel nc-χ variant needs the noise σ (pitfall #15: fail-closed)."""
    with pytest.raises(ValueError, match="sigma"):
        DynamicDPSSampler(model=_ScoreModelStub(), guidance_mode="ncchi", sigma=None)


@pytest.mark.parametrize("mode", _DDPS_MODES)
def test_dynamic_dps_mode_improves_consistency(mode: str) -> None:
    """Each variant's dynamic guidance pulls the estimate toward the measurement."""
    measurement, mask, shape = _undersampled_measurement()
    out = _ddps(mode).sample(measurement, mask)
    assert out.shape == shape and torch.isfinite(out).all()
    resid_out = ((fft2c(out) - measurement) * mask).abs().sum()
    resid_zero = (measurement * mask).abs().sum()
    assert resid_out < resid_zero, f"{mode} did not improve consistency"


def _sample_seeded(sampler, measurement, mask, seed: int = 123):
    """Sample with a fixed RNG so the only difference across runs is the mode
    (sample() draws a fresh random x_T internally)."""
    torch.manual_seed(seed)
    return sampler.sample(measurement, mask)


def test_dynamic_dps_modes_are_distinct() -> None:
    """The three variants are genuinely different guidance mechanisms (same x_T)."""
    measurement, mask, _ = _undersampled_measurement()
    outs = {m: _sample_seeded(_ddps(m), measurement, mask) for m in _DDPS_MODES}
    assert not torch.allclose(outs["adaptive_step"], outs["pigdm"])
    # ncchi reduces to adaptive_step only as σ→0; at σ=0.01 it differs.
    assert not torch.allclose(outs["adaptive_step"], outs["ncchi"])


def test_dynamic_dps_ncchi_reduces_to_adaptive_at_zero_noise() -> None:
    """Corner case: σ→0 ⇒ nc-χ floor vanishes ⇒ identical to adaptive_step."""
    measurement, mask, _ = _undersampled_measurement()
    out_adaptive = _sample_seeded(_ddps("adaptive_step"), measurement, mask)
    out_ncchi0 = _sample_seeded(_ddps("ncchi", sigma=1e-9), measurement, mask)
    assert torch.allclose(out_adaptive, out_ncchi0, atol=1e-5)


def test_dps_sampler_runs_short_chain() -> None:
    sampler = DPSSampler(n_steps=3)
    shape = (1, 1, 8, 8)
    alphas = torch.linspace(0.99, 0.01, 3)

    def score_fn(x, t):
        return torch.randn_like(x) * 0.1

    def forward_op(x):
        return x

    y = torch.randn(*shape)
    out = sampler.sample(score_fn, forward_op, y, shape, alphas, torch.device("cpu"))
    assert out.shape == shape
    assert torch.isfinite(out).all()


def test_pi_gdm_sampler_runs_short_chain() -> None:
    sampler = PiGDMSampler(n_steps=3)
    shape = (1, 1, 8, 8)
    alphas = torch.linspace(0.99, 0.01, 3)

    def score_fn(x, t):
        return torch.randn_like(x) * 0.1

    out = sampler.sample(
        score_fn,
        forward_op=lambda x: x,
        adjoint_op=lambda y: y,
        y=torch.randn(*shape),
        shape=shape,
        alphas_cumprod=alphas,
        device=torch.device("cpu"),
    )
    assert out.shape == shape
    assert torch.isfinite(out).all()


def test_dds_sampler_runs_short_chain() -> None:
    sampler = DDSSampler(n_steps=3, cg_iters=2)
    shape = (1, 1, 8, 8)
    alphas = torch.linspace(0.99, 0.01, 3)

    def score_fn(x, t):
        return torch.randn_like(x) * 0.1

    out = sampler.sample(
        score_fn,
        forward_op=lambda x: x,
        adjoint_op=lambda y: y,
        y=torch.randn(*shape),
        shape=shape,
        alphas_cumprod=alphas,
        device=torch.device("cpu"),
    )
    assert out.shape == shape
    assert torch.isfinite(out).all()


def test_sampler_factory_resolves_aliases() -> None:
    """Aliases registered alongside canonical names resolve correctly."""
    s1 = get_sampler("dps_posterior", n_steps=2)
    s2 = get_sampler("DPSPosterior", n_steps=2)
    assert type(s1) is type(s2)


@pytest.mark.parametrize("mode", ["adaptive_step", "ncchi"])
def test_dynamic_dps_guidance_step_independent_of_batch(mode: str) -> None:
    """#2367: a batch of identical copies takes the same step as one image."""
    torch.manual_seed(0)
    x = torch.rand(1, 1, 16, 16)
    m = torch.zeros(1, 1, 16, 16)
    m[..., ::4] = 1.0

    def fwd(v):
        return m * fft2c(v)

    def adj(k):
        return ifft2c(m * k).real

    y = fwd(x)
    x0 = x + 0.05 * torch.randn_like(x)
    s = DynamicDPSSampler(model=_ScoreModelStub(), guidance_mode=mode, sigma=1e-3, n_coils=1)
    ab = torch.tensor(0.5)
    d1 = s._guide(x0, fwd, adj, y, ab) - x0
    xb, yb = x0.repeat(8, 1, 1, 1), y.repeat(8, 1, 1, 1)
    d8 = s._guide(xb, fwd, adj, yb, ab) - xb
    for b in range(8):
        torch.testing.assert_close(d8[b], d1[0], rtol=1e-5, atol=1e-6)
