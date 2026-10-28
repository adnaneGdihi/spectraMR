"""Unit tests for :class:`BlochManifoldDPSSampler` (Idea 2 / B-2).

Direct-import tests covering:
- registry membership + capabilities,
- reverse-sampling shape preservation + no-NaN,
- projection lowers the manifold residual of a Tweedie estimate,
- the DPS likelihood gradient is non-zero and scales with zeta,
- zeta=0 + no-projection degenerate run (plain DDIM),
- multi-coil SENSE path,
- deterministic seeding.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from spectramr.infrastructure.physics.fft_ops import fft2c, sense_forward
from spectramr.models.diffusion.bloch_manifold_dps_sampler import (
    BlochManifoldDPSSampler,
    rescaled_linear_betas,
)
from spectramr.models.physics.bloch_manifold_projector import BlochManifoldProjector
from spectramr.models.registry import MODEL_REGISTRY


class _DummyEps(nn.Module):
    """A tiny complex epsilon-predictor: a single conv over real/imag channels."""

    def __init__(self, channels: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2 * channels, 2 * channels, 3, padding=1)
        self.channels = channels

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor | None = None) -> torch.Tensor:
        r = torch.view_as_real(x).permute(0, 1, 4, 2, 3)
        r = r.reshape(x.shape[0], 2 * self.channels, x.shape[2], x.shape[3])
        y = self.conv(r)
        y = y.reshape(x.shape[0], self.channels, 2, x.shape[2], x.shape[3])
        return torch.complex(y[:, :, 0], y[:, :, 1])


class _NearIdentity(nn.Module):
    """A score net that predicts ~zero noise => Tweedie estimate ~ x_t/sqrt(ab)."""

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor | None = None) -> torch.Tensor:
        return torch.zeros_like(x)


def _make_sampler(score=None, **kw) -> BlochManifoldDPSSampler:
    score = score or _DummyEps()
    proj = BlochManifoldProjector(n_inner_iterations=3)
    defaults = {"sampling_steps": 50, "sampler_type": "ddim", "likelihood_weight_zeta": 0.5}
    defaults.update(kw)
    return BlochManifoldDPSSampler(score, projector=proj, **defaults)


def test_registered_with_capabilities():
    assert "bloch_manifold_dps_sampler" in MODEL_REGISTRY
    entry = MODEL_REGISTRY["bloch_manifold_dps_sampler"]
    assert entry["mode"] == "diffusion"
    caps = entry["capabilities"]
    assert caps.accepts_complex is True
    assert caps.input_domain == "kspace"
    assert caps.output_domain == "image"


def test_sample_shape_and_no_nan():
    s = _make_sampler()
    b, c, h, w = 2, 1, 8, 8
    img = torch.complex(torch.rand(b, c, h, w) * 0.05, torch.rand(b, c, h, w) * 0.05)
    y = fft2c(img)
    g = torch.Generator().manual_seed(0)
    out = s.sample(y, shape=(b, c, h, w), num_steps=6, generator=g)
    assert out.shape == (b, c, h, w)
    assert torch.is_complex(out)
    assert torch.isfinite(out.real).all() and torch.isfinite(out.imag).all()


def test_projection_lowers_manifold_residual():
    """The projection must not increase a Tweedie estimate's manifold residual."""
    s = _make_sampler()
    torch.manual_seed(1)
    x0 = torch.complex(torch.rand(1, 1, 8, 8) * 0.05 + 0.01, torch.rand(1, 1, 8, 8) * 0.05)
    before = float(s.manifold_residual(x0))
    x0_proj = s.project(x0)
    after = float(s.projector.residual(x0_proj).mean())
    assert after <= before + 1e-6


def test_likelihood_gradient_nonzero_and_scales_with_zeta():
    score = _DummyEps()
    proj = BlochManifoldProjector(n_inner_iterations=2)
    b, c, h, w = 1, 1, 8, 8
    img = torch.complex(torch.rand(b, c, h, w) * 0.05, torch.rand(b, c, h, w) * 0.05)
    y = fft2c(img)
    x_t = torch.complex(torch.randn(b, c, h, w), torch.randn(b, c, h, w))

    def x0_fn(z, ti=10):
        tt = torch.full((z.shape[0],), ti, dtype=torch.long)
        return s1.tweedie_x0(z, tt, ti)

    s1 = BlochManifoldDPSSampler(score, projector=proj, sampling_steps=50, likelihood_weight_zeta=1.0)
    g1 = s1.likelihood_gradient(x_t, x0_fn, y, None, None)
    assert torch.is_complex(g1)
    assert float(g1.abs().sum()) > 0  # non-trivial guidance
    s2 = BlochManifoldDPSSampler(score, projector=proj, sampling_steps=50, likelihood_weight_zeta=2.0)
    g2 = s2.likelihood_gradient(x_t, x0_fn, y, None, None)
    # The DPS gradient is linear in zeta.
    assert torch.allclose(g2, 2.0 * g1, atol=1e-5)


def test_zeta_zero_and_no_projection_runs_ddim():
    s = _make_sampler(likelihood_weight_zeta=0.0, projection_every=10**9)
    b, c, h, w = 1, 1, 8, 8
    img = torch.complex(torch.rand(b, c, h, w) * 0.05, torch.rand(b, c, h, w) * 0.05)
    y = fft2c(img)
    g = torch.Generator().manual_seed(0)
    out = s.sample(y, shape=(b, c, h, w), num_steps=4, generator=g)
    assert torch.isfinite(out.real).all()
    # zeta=0 => the likelihood gradient is exactly zero.
    z = s.likelihood_gradient(
        torch.complex(torch.randn(b, c, h, w), torch.randn(b, c, h, w)),
        lambda v: v, y, None, None,
    )
    assert float(z.abs().sum()) == 0.0


def test_multicoil_sense_path():
    s = _make_sampler(forward_operator="sense")
    b, c, h, w, ncoil = 1, 1, 8, 8, 4
    img = torch.complex(torch.rand(b, c, h, w) * 0.05, torch.rand(b, c, h, w) * 0.05)
    smaps = torch.complex(torch.rand(b, ncoil, h, w), torch.rand(b, ncoil, h, w))
    y = sense_forward(img, smaps=smaps)
    out = s.sample(y, shape=(b, c, h, w), coil_sensitivities=smaps, num_steps=4)
    assert out.shape == (b, c, h, w)
    assert torch.isfinite(out.real).all()


def test_deterministic_seeding():
    s = _make_sampler(sampler_type="ddpm")
    b, c, h, w = 1, 1, 8, 8
    img = torch.complex(torch.rand(b, c, h, w) * 0.05, torch.rand(b, c, h, w) * 0.05)
    y = fft2c(img)
    g1 = torch.Generator().manual_seed(123)
    out1 = s.sample(y, shape=(b, c, h, w), num_steps=5, generator=g1)
    g2 = torch.Generator().manual_seed(123)
    out2 = s.sample(y, shape=(b, c, h, w), num_steps=5, generator=g2)
    assert torch.allclose(out1, out2, atol=1e-6)


def test_invalid_args_raise():
    import pytest

    score = _DummyEps()
    with pytest.raises(ValueError):
        BlochManifoldDPSSampler(score, sampler_type="bogus")
    with pytest.raises(ValueError):
        BlochManifoldDPSSampler(score, prediction_type="bogus")
    with pytest.raises(ValueError):
        BlochManifoldDPSSampler(score, likelihood_weight_zeta=-1.0)


def test_trace_returned_and_finite():
    s = _make_sampler()
    b, c, h, w = 1, 1, 8, 8
    img = torch.complex(torch.rand(b, c, h, w) * 0.05, torch.rand(b, c, h, w) * 0.05)
    y = fft2c(img)
    g = torch.Generator().manual_seed(0)
    _out, trace = s.sample(y, shape=(b, c, h, w), num_steps=4, generator=g, return_trace=True)
    assert len(trace) >= 1
    assert all(t == t for t in trace)  # no NaN


def test_the_schedule_stays_finite_on_a_short_chain():
    """PLANTED (#2441): unclamped, beta_end = 20/T reaches 1 at T <= 20 and 1/sqrt(abar) is inf."""
    for horizon in (20, 10, 4):
        abar = torch.cumprod(1.0 - rescaled_linear_betas(horizon), dim=0)
        assert (abar > 0).all()
        assert torch.isfinite(1.0 / abar.sqrt()).all()


def test_the_schedule_is_the_reference_one_where_the_clamp_does_not_bite():
    reference = torch.linspace(1e-4, 2e-2, 1000, dtype=torch.float32)
    assert torch.equal(rescaled_linear_betas(1000), reference)


class _StepSpy(_NearIdentity):
    def __init__(self) -> None:
        super().__init__()
        self.steps: list[int] = []

    def forward(self, x, timesteps=None):
        self.steps.append(int(timesteps[0]))
        return super().forward(x, timesteps)


def test_the_reverse_loop_strides_the_trained_horizon():
    """PLANTED (#2441): the schedule and step indices spanned sampling_steps, not the trained T."""
    spy = _StepSpy()
    s = _make_sampler(spy, sampling_steps=5, num_timesteps=40, likelihood_weight_zeta=0.0)
    assert s.alphas_cumprod.shape == (40,)
    torch.testing.assert_close(s.alphas_cumprod, torch.cumprod(1.0 - rescaled_linear_betas(40), 0))
    y = fft2c(torch.complex(torch.rand(1, 1, 8, 8), torch.rand(1, 1, 8, 8)) * 0.05)
    s.sample(y, shape=(1, 1, 8, 8), generator=torch.Generator().manual_seed(0))
    assert max(spy.steps) == 39 and min(spy.steps) == 0


def test_more_reverse_steps_than_the_trained_horizon_raises():
    with pytest.raises(ValueError, match="sampling_steps=50 must lie in"):
        _make_sampler(sampling_steps=50, num_timesteps=29)
