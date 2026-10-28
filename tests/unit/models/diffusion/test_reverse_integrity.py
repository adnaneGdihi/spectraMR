"""Reverse-loop integrity verifier (``model_kwargs.reverse_integrity_check``).

One planted violation per invariant, and per shape the invariant can take
(non-negotiable 15), then the loops: each reverse mode with the check on must
verify every executed step on a well-behaved model, and the loops that promise a
magnitude bound must report a model that breaks one rather than pass it through.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.diffusion.kspace_process import (
    VALID_REVERSE_MODES,
    KSpaceUndersamplingProcess,
    PhysicsInformedColdDiffusion,
)
from spectramr.models.diffusion.reverse_integrity import (
    ReverseIntegrityError,
    validate_integrity_check,
    verify_coverage,
    verify_monotone,
    verify_observed_discrepancy,
    verify_partition,
    verify_state,
)

pytestmark = pytest.mark.unit

B, C, H, W = 1, 2, 16, 16


def _state():
    meas = torch.zeros(B, C, H, W)
    obs = torch.zeros(B, 1, H, W)
    obs[..., ::4, :] = 1.0
    meas = torch.randn(B, C, H, W) * obs
    ceil = torch.full((B, 1, 1, 1), 10.0)
    return meas.clone(), meas, obs, ceil


def _raises(invariant: str, fn, *args, **kwargs) -> None:
    with pytest.raises(ReverseIntegrityError) as info:
        fn(*args, **kwargs)
    assert info.value.invariant == invariant


def test_a_clean_state_passes() -> None:
    x, meas, obs, ceil = _state()
    verify_state(x, meas, obs, ceil=ceil, dc_hard=True, where="t")


def test_shape_mismatch_is_refused() -> None:
    x, meas, obs, ceil = _state()
    _raises("shape", verify_state, x[..., :-1], meas, obs, ceil=ceil, dc_hard=True, where="t")


def test_an_odd_channel_count_is_refused() -> None:
    _, _, obs, ceil = _state()
    x = torch.zeros(B, 3, H, W)
    _raises("shape", verify_state, x, x.clone(), obs, ceil=ceil, dc_hard=False, where="t")


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_is_refused(bad) -> None:
    x, meas, obs, _ = _state()
    x[0, 0, 1, 1] = bad
    _raises("finite", verify_state, x, meas, obs, ceil=None, dc_hard=False, where="t")


def test_a_moved_measured_bin_is_refused_under_hard_dc_only() -> None:
    x, meas, obs, ceil = _state()
    x[0, 1, 0, 3] += 0.5  # row 0 is measured
    _raises("observed_pinned", verify_state, x, meas, obs, ceil=ceil, dc_hard=True, where="t")
    verify_state(x, meas, obs, ceil=ceil, dc_hard=False, where="t")


def test_an_unmeasured_bin_above_the_ceiling_is_refused() -> None:
    x, meas, obs, ceil = _state()
    x[0, 0, 1, 1] = 11.0  # row 1 is unmeasured
    _raises("magnitude_ceiling", verify_state, x, meas, obs, ceil=ceil, dc_hard=True, where="t")


def test_the_ceiling_is_on_the_complex_modulus_not_each_channel() -> None:
    x, meas, obs, ceil = _state()
    x[0, 0, 1, 1] = 8.0
    x[0, 1, 1, 1] = 8.0  # each channel under 10, modulus 11.3
    _raises("magnitude_ceiling", verify_state, x, meas, obs, ceil=ceil, dc_hard=True, where="t")


def test_measured_bins_are_exempt_from_the_ceiling() -> None:
    x, meas, obs, _ = _state()
    ceil = torch.full((B, 1, 1, 1), 1e-3)
    verify_state(x, meas, obs, ceil=ceil, dc_hard=True, where="t")


def test_a_reveal_overlapping_a_committed_bin_is_refused() -> None:
    obs = torch.zeros(B, 1, H, W)
    committed = torch.zeros_like(obs)
    committed[..., 2, :] = 1.0
    reveal = torch.zeros_like(obs)
    reveal[..., 2:4, :] = 1.0
    _raises("partition", verify_partition, reveal, committed, obs, where="t")


def test_a_reveal_overlapping_a_measured_bin_is_refused() -> None:
    _, _, obs, _ = _state()
    reveal = torch.zeros_like(obs)
    reveal[..., 0, :] = 1.0
    _raises("partition", verify_partition, reveal, torch.zeros_like(obs), obs, where="t")


def test_an_unwritten_bin_after_the_loop_is_refused() -> None:
    _, _, obs, _ = _state()
    written = 1.0 - obs
    written[..., 1, 5] = 0.0
    _raises("coverage", verify_coverage, written, obs, where="t")
    written[..., 1, 5] = 1.0
    verify_coverage(written, obs, where="t")


def test_a_measured_bin_inside_the_trust_disc_passes() -> None:
    x, meas, obs, _ = _state()
    x = x + 0.05 * obs
    verify_observed_discrepancy(x, meas, obs, torch.full((B, 1, 1, 1), 0.1), log_scaled=False, where="t")


def test_a_measured_bin_outside_the_trust_disc_is_refused() -> None:
    x, meas, obs, _ = _state()
    x[0, 0, 0, 3] += 0.5
    radius = torch.full((B, 1, 1, 1), 0.1)
    _raises("observed_discrepancy", verify_observed_discrepancy, x, meas, obs, radius, log_scaled=False, where="t")


def test_the_trust_disc_is_on_the_complex_modulus() -> None:
    """0.08 on each of Re and Im is 0.113 in modulus, outside a 0.1 disc."""
    x, meas, obs, _ = _state()
    x[0, 0, 0, 3] += 0.08
    x[0, 1, 0, 3] += 0.08
    radius = torch.full((B, 1, 1, 1), 0.1)
    _raises("observed_discrepancy", verify_observed_discrepancy, x, meas, obs, radius, log_scaled=False, where="t")


def test_the_trust_disc_ignores_unmeasured_bins() -> None:
    x, meas, obs, _ = _state()
    x[0, 0, 1, 3] += 100.0
    verify_observed_discrepancy(x, meas, obs, torch.full((B, 1, 1, 1), 0.1), log_scaled=False, where="t")


def test_a_non_increasing_sequence_passes() -> None:
    values = [torch.tensor([3.0, 2.0]), torch.tensor([2.0, 2.0]), torch.tensor([1.0, 2.0])]
    verify_monotone(values, invariant="projection_residual", where="t", rel_tol=0.0)


@pytest.mark.parametrize("at", [1, 2])
def test_a_rise_in_any_sample_at_any_iteration_is_refused(at) -> None:
    values = [torch.tensor([3.0, 2.0]), torch.tensor([2.0, 2.0]), torch.tensor([1.0, 2.0])]
    values[at] = values[at] + torch.tensor([0.0, 1.0])
    with pytest.raises(ReverseIntegrityError, match="iteration") as info:
        verify_monotone(values, invariant="projection_residual", where="t", rel_tol=0.0)
    assert info.value.invariant == "projection_residual"


def test_a_rise_within_the_declared_tolerance_passes() -> None:
    values = [torch.tensor([1.0]), torch.tensor([1.005])]
    verify_monotone(values, invariant="projection_residual", where="t", rel_tol=1e-2)
    with pytest.raises(ReverseIntegrityError):
        verify_monotone(values, invariant="projection_residual", where="t", rel_tol=1e-3)


@pytest.mark.parametrize("value", ["warn", "on", True, 1])
def test_an_unknown_setting_raises(value) -> None:
    with pytest.raises(ValueError, match="reverse_integrity_check"):
        validate_integrity_check(value)


def test_absent_means_off() -> None:
    assert validate_integrity_check(None) == "off"
    assert validate_integrity_check("raise") == "raise"


# ---------------------------------------------------------------------------
# The loops.
# ---------------------------------------------------------------------------


class _Model(torch.nn.Module):
    """Returns ``gain * target`` everywhere; ``gain = 1`` is the oracle."""

    def __init__(self, target: torch.Tensor, gain: float = 1.0) -> None:
        super().__init__()
        self.target, self.gain = target, gain
        self.kspace_process = KSpaceUndersamplingProcess(
            num_timesteps=8,
            base_acceleration=1.0,
            max_acceleration=4.0,
            center_fraction=0.08,
            train_identity_rung=True,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, **_kw: object) -> torch.Tensor:
        return self.target * self.gain


def _sampler(mode: str, model: _Model, check: str = "raise") -> PhysicsInformedColdDiffusion:
    return PhysicsInformedColdDiffusion(
        model=model,
        num_timesteps=8,
        max_acceleration=4.0,
        center_fraction=0.08,
        dc_method="hard",
        reverse_mode=mode,
        reverse_clip_ratio=1.5,
        refine_alpha=0.5 if mode == "reveal_refine_dc" else None,
        support_radius=2 if mode == "support_ordered_dc" else None,
        kspace_log_scaled=False,
        integrity_check=check,
    )


def _problem(gain: float = 1.0):
    torch.manual_seed(0)
    target = torch.randn(B, C, 32, 32)
    model = _Model(target, gain)
    mask = model.kspace_process.mask_at(7, (32, 32)).reshape(1, 1, 32, 32).float()
    return model, target * mask, mask


@pytest.mark.parametrize("mode", sorted(VALID_REVERSE_MODES - {"additive"}))
def test_every_bounded_loop_verifies_each_executed_step(mode) -> None:
    model, meas, mask = _problem()
    sampler = _sampler(mode, model)
    sampler.sample(meas, mask, start_timestep=7)
    stats = sampler.reverse_stats
    assert stats["integrity_check"] == "raise"
    assert stats["integrity_steps_verified"] == stats["effective_steps"] > 0


@pytest.mark.parametrize("mode", sorted(VALID_REVERSE_MODES))
def test_the_check_off_verifies_nothing(mode) -> None:
    model, meas, mask = _problem()
    sampler = _sampler(mode, model, check="off")
    sampler.sample(meas, mask, start_timestep=7)
    assert sampler.reverse_stats["integrity_steps_verified"] == 0


def test_plain_additive_reports_the_first_overshoot() -> None:
    """``additive`` has no bound; with the check on it stops at the first step past the ceiling."""
    model, meas, mask = _problem(gain=50.0)
    with pytest.raises(ReverseIntegrityError) as info:
        _sampler("additive", model).sample(meas, mask, start_timestep=7)
    assert info.value.invariant == "magnitude_ceiling"


@pytest.mark.parametrize("mode", ["additive_normalized", "replace_freeze_dc", "reveal_refine_dc"])
def test_a_non_finite_model_is_stopped_at_the_step_that_wrote_it(mode) -> None:
    model, meas, mask = _problem()
    model.target = model.target.clone()
    model.target[0, 0, 5, 5] = float("nan")
    with pytest.raises(ReverseIntegrityError) as info:
        _sampler(mode, model).sample(meas, mask, start_timestep=7)
    assert info.value.invariant in {"finite", "magnitude_ceiling"}


def test_the_counter_resets_on_every_sample_call() -> None:
    model, meas, mask = _problem()
    sampler = _sampler("replace_freeze_dc", model)
    sampler.sample(meas, mask, start_timestep=7)
    first = sampler.last_integrity_steps_verified
    sampler.sample(meas, mask, start_timestep=7)
    assert sampler.last_integrity_steps_verified == first
