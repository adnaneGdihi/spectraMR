"""Normalized additive reverse loop (``reverse_sampling_mode: additive_normalized``).

The planted pairs (non-negotiable 15):

* an estimate that overshoots by 50x drives plain ``additive`` far past the
  ceiling and leaves ``additive_normalized`` under it;
* a model whose every call predicts a different constant leaves a
  reveal-step coefficient at the MEAN of its two writes -- a loop that kept the
  last write (or summed them) fails.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.diffusion.kspace_process import (
    KSpaceUndersamplingProcess,
    PhysicsInformedColdDiffusion,
    paired_magnitude,
)
from spectramr.models.diffusion.reveal_attribution import PARTITIONED_REVERSE_MODES
from spectramr.models.diffusion.reverse_additive_normalized import (
    NORMALIZED_ADDITIVE_REVERSE_MODE,
    additive_increment,
)

pytestmark = pytest.mark.unit

B, C, H, W = 1, 2, 32, 32
T = 8
HEAD = 7


def _process() -> KSpaceUndersamplingProcess:
    return KSpaceUndersamplingProcess(
        num_timesteps=T,
        base_acceleration=1.0,
        max_acceleration=4.0,
        center_fraction=0.08,
        train_identity_rung=True,
    )


class _Scaled(torch.nn.Module):
    """Predicts ``gain * target``; ``gain = 1`` is the oracle."""

    def __init__(self, target: torch.Tensor, gain: float = 1.0) -> None:
        super().__init__()
        self.target, self.gain = target, gain
        self.kspace_process = _process()

    def forward(self, x: torch.Tensor, t: torch.Tensor, **_kw: object) -> torch.Tensor:
        return self.target * self.gain


class _PerCall(torch.nn.Module):
    """Predicts the constant ``call index + 1`` everywhere."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.kspace_process = _process()

    def forward(self, x: torch.Tensor, t: torch.Tensor, **_kw: object) -> torch.Tensor:
        self.calls += 1
        return torch.full_like(x, float(self.calls))


def _sampler(model, mode=NORMALIZED_ADDITIVE_REVERSE_MODE, ratio=1.5, **kw):
    return PhysicsInformedColdDiffusion(
        model=model,
        num_timesteps=T,
        max_acceleration=4.0,
        center_fraction=0.08,
        dc_method="hard",
        reverse_mode=mode,
        reverse_clip_ratio=ratio,
        kspace_log_scaled=False,
        integrity_check="raise" if mode == NORMALIZED_ADDITIVE_REVERSE_MODE else "off",
        **kw,
    )


def _mask(process: KSpaceUndersamplingProcess) -> torch.Tensor:
    return process.mask_at(HEAD, (H, W)).reshape(1, 1, H, W).float()


def test_the_mode_is_not_partitioned() -> None:
    """Every coefficient is written twice, so no single step owns it."""
    assert NORMALIZED_ADDITIVE_REVERSE_MODE not in PARTITIONED_REVERSE_MODES


def test_the_oracle_is_reproduced_exactly() -> None:
    torch.manual_seed(0)
    target = torch.randn(B, C, H, W)
    model = _Scaled(target)
    mask = _mask(model.kspace_process)
    out = _sampler(model, ratio=100.0).sample(target * mask, mask, start_timestep=HEAD)
    assert torch.allclose(out, target, atol=1e-5)


def test_an_overshooting_model_stays_under_the_ceiling() -> None:
    torch.manual_seed(0)
    target = torch.randn(B, C, H, W)
    mask = _mask(_process())
    meas = target * mask
    ceil = 1.5 * float(paired_magnitude(meas).max())

    plain = _sampler(_Scaled(target, 50.0), mode="additive").sample(meas, mask, start_timestep=HEAD)
    normalized = _sampler(_Scaled(target, 50.0)).sample(meas, mask, start_timestep=HEAD)

    assert float(paired_magnitude(plain).max()) > 10 * ceil
    assert float(paired_magnitude(normalized).max()) <= ceil * (1 + 1e-4)
    assert torch.equal(normalized * mask, meas), "measured lines must stay pinned"


def test_a_coefficient_holds_the_mean_of_its_writes() -> None:
    model = _PerCall()
    mask = _mask(model.kspace_process)
    meas = 100.0 * torch.ones(B, C, H, W) * mask
    sampler = _sampler(model, ratio=100.0)
    out = sampler.sample(meas, mask, start_timestep=HEAD)

    schedule = sampler.last_schedule
    obs = sampler._observed_support(meas, mask, 1e-8)
    last = float(len(schedule))
    expected = meas.clone()
    early = torch.zeros_like(obs)
    for i in range(len(schedule) - 1):
        inc = additive_increment(sampler, i, schedule, obs, meas)
        expected = expected + inc * (float(i + 1) + last) / 2.0
        early = early + inc
    expected = expected + (1.0 - obs - early) * last

    assert bool(early.any()), "the ladder must reveal something before the terminal call"
    assert torch.allclose(out, expected), "each write must be averaged, not kept or summed"


def test_a_strided_schedule_still_covers_every_coefficient() -> None:
    """``check_coverage`` raises under ``raise`` if a stride skipped a rung's lines."""
    torch.manual_seed(0)
    target = torch.randn(B, C, H, W)
    model = _Scaled(target)
    mask = _mask(model.kspace_process)
    sampler = _sampler(model, ratio=100.0, sampling_steps=3)
    out = sampler.sample(target * mask, mask, start_timestep=HEAD)
    assert len(sampler.last_schedule) < HEAD + 1
    assert bool((paired_magnitude(out) > 0).all())


def test_the_stats_report_every_call_and_the_largest_divisor() -> None:
    torch.manual_seed(0)
    target = torch.randn(B, C, H, W)
    model = _Scaled(target, 50.0)
    mask = _mask(model.kspace_process)
    sampler = _sampler(model)
    sampler.sample(target * mask, mask, start_timestep=HEAD)
    stats = sampler.reverse_stats
    assert stats["effective_steps"] == len(stats["schedule"])
    assert stats["skipped_steps"] == 0
    assert stats["terminal_timestep_called"] == stats["schedule"][-1]
    assert isinstance(stats["band_divisor_max"], float)
    assert stats["band_divisor_max"] > 1.0
    assert stats["integrity_steps_verified"] == stats["effective_steps"]


def test_the_other_modes_publish_no_divisor() -> None:
    torch.manual_seed(0)
    target = torch.randn(B, C, H, W)
    model = _Scaled(target)
    mask = _mask(model.kspace_process)
    sampler = _sampler(model, mode="replace_freeze_dc")
    sampler.sample(target * mask, mask, start_timestep=HEAD)
    assert sampler.reverse_stats["band_divisor_max"] is None


def test_a_band_local_ceiling_never_moves_an_acquired_line() -> None:
    """The band-local ceiling of an outer band can sit below its own measured peak (#2310)."""
    torch.manual_seed(0)
    target = torch.randn(B, C, H, W)
    model = _Scaled(target)
    mask = _mask(model.kspace_process)
    meas = target * mask
    ij = mask[0, 0].nonzero()
    r, c = ij[(ij - torch.tensor([H // 2, W // 2])).pow(2).sum(1).argmax()].tolist()
    meas[..., r, c] = 100.0 * meas[..., r, c] + 100.0
    sampler = _sampler(model, clip_reference="band_local")
    out = sampler.sample(meas, mask, start_timestep=HEAD)
    assert torch.equal(out * mask, meas)


def test_hard_dc_keeps_an_outer_band_above_the_envelope_pinned() -> None:
    """PLANTED (#2310): the ceiling clamp ran after hard DC on the whole plane."""
    from tests.unit.models.diffusion import _reverse_phantom as phantom

    out, meas, acquired = phantom.band_local_outlier_run("additive_normalized")
    assert torch.equal(out * acquired, meas)


def test_train_mode_reconstructs_what_eval_mode_does() -> None:
    """PLANTED (#2471): each step's contribution came from ``q_sample``, a dynamic training mask in
    train mode, so the reconstruction changed with the module's mode."""
    from tests.unit.models.diffusion import _reverse_phantom as phantom

    outs, _ = phantom.module_mode_runs(NORMALIZED_ADDITIVE_REVERSE_MODE)
    assert all(torch.equal(out, outs[0]) for out in outs[1:])
