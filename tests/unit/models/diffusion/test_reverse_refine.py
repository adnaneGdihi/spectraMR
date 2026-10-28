"""Reveal-then-refine reverse loop (``reverse_sampling_mode: reveal_refine_dc``).

The planted pairs (non-negotiable 15):

* ``reverse_refine_alpha`` declared under another mode -> raises (unread knob);
  omitted under the refine mode -> raises (no neutral default);
* under ``replace_freeze_dc_t0`` the first-revealed band keeps the estimate of
  the call that revealed it; under refine at ``alpha = 1`` it carries the
  terminal call's estimate. A loop that froze would pass the first and fail the
  second.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from spectramr.models.diffusion.kspace_process import (
    KSpaceUndersamplingProcess,
    PhysicsInformedColdDiffusion,
)
from spectramr.models.diffusion.reverse_refine import (
    REFINE_REVERSE_MODE,
    validate_refine_alpha,
)

pytestmark = pytest.mark.unit

B, C, H, W = 1, 2, 32, 32


class _StepValued(torch.nn.Module):
    """Predicts the constant ``t + 1`` everywhere and records each call's ``t``."""

    def __init__(self, *, identity_rung: bool = True) -> None:
        super().__init__()
        self.seen: list[int] = []
        self.kspace_process = KSpaceUndersamplingProcess(
            num_timesteps=8,
            base_acceleration=1.0,
            max_acceleration=4.0,
            center_fraction=0.08,
            train_identity_rung=identity_rung,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, **_kw: object) -> torch.Tensor:
        self.seen.append(int(t[0]))
        return torch.full_like(x, float(t[0]) + 1.0)


def _build(mode: str, alpha: float | None = None, **kw):
    model = kw.pop("model", None) or _StepValued()
    sampler = PhysicsInformedColdDiffusion(
        model=model,
        num_timesteps=8,
        max_acceleration=4.0,
        center_fraction=0.08,
        dc_method="hard",
        reverse_mode=mode,
        refine_alpha=alpha,
        kspace_log_scaled=False,
        **kw,
    )
    return sampler, model


#: Large enough that the ``t + 1`` predictions sit under the default magnitude
#: ceiling (``4 x max|observed|``), so the clamp does not mask the blend.
_LEVEL = 10.0


def _measurement(start_t: int, process: KSpaceUndersamplingProcess):
    mask = process.mask_at(start_t, (H, W)).reshape(1, 1, H, W)
    return _LEVEL * torch.ones(B, C, H, W) * mask, mask


@pytest.mark.parametrize("mode", ["additive", "replace_freeze_dc", "replace_freeze_dc_t0"])
def test_alpha_under_another_mode_raises(mode: str) -> None:
    with pytest.raises(ValueError, match="only read by"):
        validate_refine_alpha(mode, 0.5)


def test_refine_mode_without_alpha_raises() -> None:
    with pytest.raises(ValueError, match="requires"):
        validate_refine_alpha(REFINE_REVERSE_MODE, None)


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.5, math.nan])
def test_alpha_outside_unit_interval_raises(alpha: float) -> None:
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        validate_refine_alpha(REFINE_REVERSE_MODE, alpha)


def test_other_modes_resolve_to_none() -> None:
    assert validate_refine_alpha("replace_freeze_dc", None) is None
    assert validate_refine_alpha(REFINE_REVERSE_MODE, 1) == 1.0


def test_sampler_constructor_enforces_the_pairing() -> None:
    with pytest.raises(ValueError, match="only read by"):
        _build("replace_freeze_dc", 0.5)
    with pytest.raises(ValueError, match="requires"):
        _build(REFINE_REVERSE_MODE, None)


def test_refine_runs_on_a_floor_one_ladder_without_an_identity_rung() -> None:
    """Its reveal is keyed to the next rung like the other modes, so the t=0 floor
    is no longer a precondition; the terminal call runs at the floor."""
    sampler, model = _build(REFINE_REVERSE_MODE, 0.5, model=_StepValued(identity_rung=False))
    measurement, mask = _measurement(4, model.kspace_process)
    sampler.sample(measurement, mask, start_timestep=4)
    assert min(model.seen) == 1


def _first_revealed_band(sampler, measurement, mask) -> torch.Tensor:
    """Bins the first effective step reveals, from the process' own masks."""
    schedule = sampler.last_schedule
    obs = (mask > 0).float()
    first = min(i for i, t in enumerate(schedule) if t in sampler.model.seen)
    key = sampler.process.mask_at(schedule[first], (H, W)).reshape(1, 1, H, W)
    return (key * (1 - obs)).bool().expand(B, C, H, W)


def test_t0_terminal_call_authors_the_unobserved_plane_as_refine_at_alpha_one() -> None:
    """The t=0 call of replace_freeze_dc_t0 rewrites every unobserved bin, which is
    what refine does at alpha=1."""
    frozen, fmodel = _build("replace_freeze_dc_t0")
    measurement, mask = _measurement(4, fmodel.kspace_process)
    out_t0 = frozen.sample(measurement, mask, start_timestep=4)
    assert min(fmodel.seen) == 0, "the terminal call must run at t=0"
    unobserved = (mask == 0).expand(B, C, H, W)
    assert torch.allclose(out_t0[unobserved], torch.tensor(1.0))

    refine, rmodel = _build(REFINE_REVERSE_MODE, 1.0)
    out_refine = refine.sample(measurement, mask, start_timestep=4)
    assert min(rmodel.seen) == 0, "the terminal call must run at t=0"
    unobserved = (mask == 0).expand(B, C, H, W)
    assert torch.allclose(out_refine[unobserved], torch.tensor(1.0)), (
        "at alpha=1 every unobserved bin carries the terminal (t=0) estimate"
    )


def test_partial_alpha_averages_successive_estimates() -> None:
    sampler, model = _build(REFINE_REVERSE_MODE, 0.5)
    measurement, mask = _measurement(4, model.kspace_process)
    out = sampler.sample(measurement, mask, start_timestep=4)
    band = _first_revealed_band(sampler, measurement, mask)
    expected = None
    for t in model.seen:  # descending: the first call writes, later ones blend
        v = float(t) + 1.0
        expected = v if expected is None else 0.5 * expected + 0.5 * v
    assert torch.allclose(out[band], torch.tensor(expected))


def test_observed_support_stays_pinned_under_hard_dc() -> None:
    sampler, model = _build(REFINE_REVERSE_MODE, 0.5)
    measurement, mask = _measurement(4, model.kspace_process)
    out = sampler.sample(measurement, mask, start_timestep=4)
    obs = (mask > 0).expand(B, C, H, W)
    assert torch.equal(out[obs], measurement[obs])


def test_output_obeys_the_magnitude_ceiling() -> None:
    class _Huge(_StepValued):
        def forward(self, x, t, **_kw):
            super().forward(x, t)
            return torch.full_like(x, 1.0e6)

    ratio = 2.0
    sampler, model = _build(REFINE_REVERSE_MODE, 0.5, model=_Huge(), reverse_clip_ratio=ratio)
    measurement, mask = _measurement(4, model.kspace_process)
    out = sampler.sample(measurement, mask, start_timestep=4)
    modulus = out.reshape(B, C // 2, 2, H, W).pow(2).sum(2).sqrt()
    ceiling = ratio * _LEVEL * math.sqrt(2.0)
    assert float(modulus.max()) <= ceiling * (1 + 1e-5)


def test_the_skip_gate_reads_no_scalar_per_step(monkeypatch) -> None:
    """The gate read ``bool(reveal.any())`` and ``bool(committed.any())`` every step."""
    import sys

    sampler, model = _build(REFINE_REVERSE_MODE, 0.5)
    measurement, mask = _measurement(4, model.kspace_process)
    expected = sampler.sample(measurement, mask, start_timestep=4)
    stats = dict(sampler.reverse_stats)
    for name in ("item", "__bool__"):
        original = getattr(torch.Tensor, name)

        def read(tensor, *args, _name=name, _original=original, **kwargs):
            if sys._getframe(1).f_code.co_filename.endswith("reverse_refine.py"):
                raise AssertionError(f"per-step host read {_name} in reverse_refine.py")
            return _original(tensor, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, name, read)
    assert torch.equal(sampler.sample(measurement, mask, start_timestep=4), expected)
    assert sampler.reverse_stats["skipped_steps"] == stats["skipped_steps"]


def test_stats_count_every_call_after_the_first_reveal() -> None:
    sampler, model = _build(REFINE_REVERSE_MODE, 0.5)
    measurement, mask = _measurement(4, model.kspace_process)
    sampler.sample(measurement, mask, start_timestep=4)
    stats = sampler.reverse_stats
    assert stats["effective_steps"] == len(model.seen)
    assert stats["effective_steps"] + stats["skipped_steps"] == len(stats["schedule"])
    assert stats["terminal_timestep_called"] == 0


def test_reveal_attribution_refuses_the_refine_mode() -> None:
    """Coefficients are rewritten, so no step 'wrote' a bin; the audit must refuse."""
    from spectramr.infrastructure.validation.config_health_checker import (
        ConfigHealthChecker,
    )
    from spectramr.models.diffusion.reveal_attribution import PARTITIONED_REVERSE_MODES

    assert REFINE_REVERSE_MODE not in PARTITIONED_REVERSE_MODES
    config = SimpleNamespace(
        validation=SimpleNamespace(sampling=SimpleNamespace(reveal_attribution=True)),
        model=SimpleNamespace(
            model_kwargs={"reverse_sampling_mode": REFINE_REVERSE_MODE},
            output_type="kspace",
        ),
        undersampling=SimpleNamespace(acceleration_type="density_nested", mask_direction="phase"),
    )
    checker = ConfigHealthChecker.__new__(ConfigHealthChecker)
    result = checker.check_reveal_attribution_preconditions(config)
    assert not result.passed
    assert REFINE_REVERSE_MODE in result.message


def test_hard_dc_keeps_an_outer_band_above_the_envelope_pinned() -> None:
    """PLANTED (#2310): the ceiling clamp ran after hard DC on the whole plane."""
    from tests.unit.models.diffusion import _reverse_phantom as phantom

    out, meas, acquired = phantom.band_local_outlier_run("reveal_refine_dc")
    assert torch.equal(out * acquired, meas)
