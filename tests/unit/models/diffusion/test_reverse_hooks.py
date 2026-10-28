"""The fixed order DC -> projection -> noise -> bound, and the clamp map each loop gets."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from spectramr.models.diffusion.kspace_process import paired_magnitude
from spectramr.models.diffusion.reverse_hooks import (
    bound_estimate,
    noise_estimator,
    prepare_reverse,
    refine_estimate,
)
from tests.unit.models.diffusion import _reverse_phantom as P

pytestmark = pytest.mark.unit

EPS = 1e-8


@pytest.fixture(scope="module")
def setting():
    proc = P.process(8.0)
    target = P.kspace()
    mask = P.mask(proc)
    return proc, target, target * mask, mask


def _prepared(setting, **kw):
    proc, target, meas, mask = setting
    sampler = P.sampler(P.Oracle(target, proc), "replace_freeze_dc", **kw)
    obs = sampler._observed_support(meas, mask, EPS)
    ctx = prepare_reverse(sampler, meas, obs, eps=EPS)
    plane = paired_magnitude(meas).shape
    measured = (obs.amax(dim=1, keepdim=True) > 0).expand(plane)
    ctx = dataclasses.replace(
        ctx, ceil=ctx.ceil.expand(plane).clone(), write_ceil=ctx.write_ceil.expand(plane).clone()
    )
    return sampler, obs, ctx, measured


def test_hard_dc_leaves_the_measured_bins_unclamped(setting) -> None:
    """The freeze loop included: its whole-plane clamp rewrote acquired bins (#2310)."""
    _, _, ctx, measured = _prepared(setting)
    assert bool(torch.isinf(ctx.write_ceil[measured]).all())
    assert torch.equal(ctx.write_ceil[~measured], ctx.ceil[~measured])


def test_soft_dc_clamps_the_measured_bins_to_the_larger_of_ceil_and_y(setting) -> None:
    _, _, meas, _ = setting
    _, _, ctx, measured = _prepared(setting, dc_method="noise_adaptive")
    own = torch.maximum(ctx.ceil, paired_magnitude(meas))
    assert torch.equal(ctx.write_ceil[measured], own[measured])


def test_fitted_trust_lifts_the_measured_clamp_so_the_disc_survives(setting) -> None:
    _, _, ctx, measured = _prepared(
        setting,
        dc_method="noise_adaptive",
        decay_envelope="per_sample",
        dc_trust="fitted_decay",
    )
    assert ctx.trust is not None and ctx.trust_radius is not None
    assert bool((ctx.write_ceil[measured] >= ctx.ceil[measured]).all())


def test_the_decay_envelope_publishes_its_slope(setting) -> None:
    sampler, _, ctx, _ = _prepared(setting, decay_envelope="per_sample")
    assert ctx.decay is not None
    assert float(sampler.last_decay_slope) == pytest.approx(float(ctx.decay.slope.mean()))
    assert int(sampler.last_decay_unfitted) == 0


def test_an_unfitted_sample_is_counted_and_its_applied_slope_is_zero(setting) -> None:
    """A noise-only slice is bounded at the noise floor with slope 0, and the column stays defined."""
    proc, target, meas, mask = setting
    noise = 1e-3 * torch.randn(meas[1:].shape, generator=torch.Generator().manual_seed(5))
    mixed = torch.cat([meas[:1], noise * mask[1:]])
    sampler, _, ctx, _ = _prepared((proc, target, mixed, mask), decay_envelope="per_sample")
    assert ctx.decay is not None
    assert int(sampler.last_decay_unfitted) == 1
    assert float(ctx.decay.slope[1]) == 0.0
    assert float(sampler.last_decay_slope) == pytest.approx(float(ctx.decay.slope[0]) / 2)


def test_the_bound_runs_after_a_projection_that_inflates(setting) -> None:
    """PLANTED: a projector that multiplies by 100 cannot push a write past the ceiling."""
    _, target, meas, _ = setting
    sampler, obs, ctx, _ = _prepared(setting)
    ctx = dataclasses.replace(ctx, projector=lambda x, o, *, where: 100.0 * x)
    x0 = refine_estimate(sampler, target, meas, obs, ctx, where="test")
    assert float(paired_magnitude(x0).max()) > 50 * float(paired_magnitude(meas).max())
    bounded = bound_estimate(sampler, x0, obs, ctx, EPS)
    assert bool((paired_magnitude(bounded) <= ctx.write_ceil * (1 + 1e-5)).all())


def test_the_noise_estimate_has_one_owner(setting) -> None:
    proc, target, _, _ = setting
    model = P.Oracle(target, proc)
    assert noise_estimator(P.sampler(model, "replace_freeze_dc")) is model.dc_layer
    del model.dc_layer
    assert noise_estimator(P.sampler(model, "replace_freeze_dc")) is not None



def test_the_envelope_bins_along_the_process_line_axis() -> None:
    """``per_sample`` fits the mask's lines: columns once the process removes columns (#2160)."""
    from spectramr.models.diffusion.kspace_process import KSpaceUndersamplingProcess

    proc = KSpaceUndersamplingProcess(
        num_timesteps=P.T,
        base_acceleration=1.0,
        max_acceleration=8.0,
        center_fraction=0.08,
        mask_type="density_nested",
        mask_direction="phase",
        phase_encode_axis=-1,
        enforce_nested=True,
        train_identity_rung=True,
        seed=42,
    )
    assert proc.line_axis == "x"
    target = P.kspace().transpose(-1, -2).contiguous()
    mask = P.mask(proc)
    meas = target * mask
    sampler = P.sampler(P.Oracle(target, proc), "replace_freeze_dc", decay_envelope="per_sample")
    obs = sampler._observed_support(meas, mask, EPS)
    ctx = prepare_reverse(sampler, meas, obs, eps=EPS)
    assert ctx.decay is not None and ctx.decay.axis.name == "column"


def test_spirit_calibrates_on_the_columns_of_a_column_mask() -> None:
    """PLANTED: SPIRiT calibrates on fully measured rows, and a column mask has none,
    so the row-only build raised; on the transposed plane it calibrates on columns."""
    from spectramr.models.diffusion.kspace_process import KSpaceUndersamplingProcess
    from spectramr.models.diffusion.reverse_spirit import build_spirit

    proc = KSpaceUndersamplingProcess(
        num_timesteps=P.T,
        base_acceleration=1.0,
        max_acceleration=8.0,
        center_fraction=0.08,
        mask_type="density_nested",
        mask_direction="phase",
        phase_encode_axis=-1,
        enforce_nested=True,
        train_identity_rung=True,
        seed=42,
    )
    target = P.kspace().transpose(-1, -2).contiguous()
    mask = P.mask(proc)
    meas = target * mask
    sampler = P.sampler(P.Oracle(target, proc), "replace_freeze_dc", projection="spirit")
    obs = sampler._observed_support(meas, mask, EPS)
    with pytest.raises(ValueError, match="rows to calibrate"):
        build_spirit(sampler, meas, obs)
    ctx = prepare_reverse(sampler, meas, obs, eps=EPS)
    out = ctx.projector(target.clone(), obs, where="test")
    assert out.shape == target.shape
    assert torch.equal(out * obs, target * obs)


def test_a_row_mask_keeps_the_row_envelope(setting) -> None:
    _, _, ctx, _ = _prepared(setting, decay_envelope="per_sample")
    assert ctx.decay.axis.name == "row"
