"""Single-call reconstruction (``reverse_sampling_mode: single_call_dc``).

Planted pair: under a multi-step loop the model is called at several rungs and
its own predictions feed later calls; under single_call_dc it is called exactly
once, at the measurement's rung, on the measurement itself.
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
from spectramr.models.diffusion.reverse_single_call import SINGLE_CALL_REVERSE_MODE

pytestmark = pytest.mark.unit

B, C, H, W = 1, 2, 32, 32
_LEVEL = 10.0


class _Recorder(torch.nn.Module):
    """Predicts ``t + 1`` everywhere; records each call's ``t`` and input."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[int] = []
        self.inputs: list[torch.Tensor] = []
        self.kspace_process = KSpaceUndersamplingProcess(
            num_timesteps=8,
            base_acceleration=1.0,
            max_acceleration=4.0,
            center_fraction=0.08,
            train_identity_rung=True,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, **_kw: object) -> torch.Tensor:
        self.seen.append(int(t[0]))
        self.inputs.append(x.clone())
        return torch.full_like(x, float(t[0]) + 1.0)


def _build(mode: str, **kw):
    model = kw.pop("model", None) or _Recorder()
    sampler = PhysicsInformedColdDiffusion(
        model=model,
        num_timesteps=8,
        max_acceleration=4.0,
        center_fraction=0.08,
        dc_method="hard",
        reverse_mode=mode,
        kspace_log_scaled=False,
        **kw,
    )
    return sampler, model


def _measurement(start_t: int, process: KSpaceUndersamplingProcess):
    mask = process.mask_at(start_t, (H, W)).reshape(1, 1, H, W)
    return _LEVEL * torch.ones(B, C, H, W) * mask, mask


def test_one_call_at_the_measurements_rung_on_the_measurement() -> None:
    sampler, model = _build(SINGLE_CALL_REVERSE_MODE)
    measurement, mask = _measurement(4, model.kspace_process)
    sampler.sample(measurement, mask, start_timestep=4)
    assert model.seen == [4]
    assert torch.equal(model.inputs[0], measurement)


def test_a_multi_step_loop_feeds_back_its_own_predictions() -> None:
    """PLANTED: the other pole, so the single-call assertions can fail."""
    sampler, model = _build("replace_freeze_dc_t0")
    measurement, mask = _measurement(4, model.kspace_process)
    sampler.sample(measurement, mask, start_timestep=4)
    assert len(model.seen) > 1
    assert not torch.equal(model.inputs[-1], measurement)


def test_unobserved_bins_carry_the_single_estimate_observed_bins_the_measurement() -> None:
    sampler, model = _build(SINGLE_CALL_REVERSE_MODE)
    measurement, mask = _measurement(4, model.kspace_process)
    out = sampler.sample(measurement, mask, start_timestep=4)
    obs = (mask > 0).expand(B, C, H, W)
    assert torch.equal(out[obs], measurement[obs])
    assert torch.allclose(out[~obs], torch.tensor(5.0))


def test_stats_report_one_call_and_the_rest_skipped() -> None:
    sampler, _ = _build(SINGLE_CALL_REVERSE_MODE)
    measurement, mask = _measurement(4, sampler.model.kspace_process)
    sampler.sample(measurement, mask, start_timestep=4)
    stats = sampler.reverse_stats
    assert stats["effective_steps"] == 1
    assert stats["effective_steps"] + stats["skipped_steps"] == len(stats["schedule"])
    assert stats["terminal_timestep_called"] == 4
    assert stats["reverse_mode"] == SINGLE_CALL_REVERSE_MODE


def test_output_obeys_the_magnitude_ceiling() -> None:
    class _Huge(_Recorder):
        def forward(self, x, t, **_kw):
            super().forward(x, t)
            return torch.full_like(x, 1.0e6)

    ratio = 2.0
    sampler, model = _build(SINGLE_CALL_REVERSE_MODE, model=_Huge(), reverse_clip_ratio=ratio)
    measurement, mask = _measurement(4, model.kspace_process)
    out = sampler.sample(measurement, mask, start_timestep=4)
    modulus = out.reshape(B, C // 2, 2, H, W).pow(2).sum(2).sqrt()
    assert float(modulus.max()) <= ratio * _LEVEL * math.sqrt(2.0) * (1 + 1e-5)


def test_refine_alpha_is_refused() -> None:
    with pytest.raises(ValueError, match="only read by"):
        _build(SINGLE_CALL_REVERSE_MODE, refine_alpha=0.5)


def test_reveal_attribution_refuses_the_single_call_mode() -> None:
    """One call writes every coefficient, so there is no per-step partition."""
    from spectramr.infrastructure.validation.config_health_checker import ConfigHealthChecker
    from spectramr.models.diffusion.reveal_attribution import PARTITIONED_REVERSE_MODES

    assert SINGLE_CALL_REVERSE_MODE not in PARTITIONED_REVERSE_MODES
    config = SimpleNamespace(
        validation=SimpleNamespace(sampling=SimpleNamespace(reveal_attribution=True)),
        model=SimpleNamespace(
            model_kwargs={"reverse_sampling_mode": SINGLE_CALL_REVERSE_MODE},
            output_type="kspace",
        ),
        undersampling=SimpleNamespace(acceleration_type="density_nested", mask_direction="phase"),
    )
    result = ConfigHealthChecker.__new__(
        ConfigHealthChecker
    ).check_reveal_attribution_preconditions(config)
    assert not result.passed and SINGLE_CALL_REVERSE_MODE in result.message


def test_hard_dc_keeps_an_outer_band_above_the_envelope_pinned() -> None:
    """PLANTED (#2310): the ceiling clamp ran after hard DC on the whole plane."""
    from tests.unit.models.diffusion import _reverse_phantom as phantom

    out, meas, acquired = phantom.band_local_outlier_run("single_call_dc")
    assert torch.equal(out * acquired, meas)
