"""What the reverse loop RAN, as a record rather than an inference (#2067).

``last_effective_steps`` / ``last_skipped_steps`` were computed by both freeze
loops and discarded, so the fact these tests pin — that under ``dc_method:
hard`` the terminal t=0 step is skipped and the model is never called at the
rung that IS R=1 — was unobservable from outside the sampler.

The planted pair the terminal stamp exists for (non-negotiable 15):

* hard DC on a ladder whose ``mask(0)`` is all-ones -> terminal step SKIPPED,
  so ``terminal_timestep_called`` is 1, not 0;
* the same trajectory under a non-hard ``dc_method`` -> nothing is skipped and
  the terminal step RUNS, so it is 0.

A stamp that reported the tail of the schedule instead of the lowest executed
step would read 0 in both cases and could not tell them apart.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.data_consistency import SoftDataConsistency
from spectramr.models.diffusion.cold_mri_sampler import ColdMRISampler
from spectramr.models.diffusion.kspace_process import (
    KSpaceUndersamplingProcess,
    PhysicsInformedColdDiffusion,
)

pytestmark = pytest.mark.unit

B, C, H, W = 1, 2, 32, 32


class _Identity(torch.nn.Module):
    """Records every timestep it is called at, so 'was it called' is observable.

    Carries a ``kspace_process`` the way a real generator does, with
    ``train_identity_rung`` on: that is what makes t=0 a genuine rung
    (``min_meaningful_timestep() == 0``, ``mask(0)`` all-ones) and so what makes
    the terminal step schedulable in the first place. Without it the floor is 1,
    t=0 is never scheduled, and the skip these tests are about cannot arise.
    """

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[int] = []
        self.kspace_process = KSpaceUndersamplingProcess(
            num_timesteps=8,
            base_acceleration=1.0,
            max_acceleration=4.0,
            center_fraction=0.08,
            train_identity_rung=True,
        )
        # Soft DC is now delegated to the layer the generator trains rather
        # than re-derived in the reverse loop, so a stand-in model has to carry
        # one for the soft path to be exercisable at all.
        self.dc_layer = SoftDataConsistency(lambda_init=0.5)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        self.seen.append(int(t[0]))
        return x


def _build(dc_method: str, reverse_mode: str) -> tuple[PhysicsInformedColdDiffusion, _Identity]:
    model = _Identity()
    diffusion = PhysicsInformedColdDiffusion(
        model=model,
        num_timesteps=8,
        max_acceleration=4.0,
        center_fraction=0.08,
        dc_method=dc_method,
        dc_weight=0.5,
        reverse_mode=reverse_mode,
        kspace_log_scaled=False,
    )
    return diffusion, model


def _measurement_and_mask() -> tuple[torch.Tensor, torch.Tensor]:
    mask = torch.zeros(B, 1, H, W)
    mask[:, :, ::2, :] = 1.0  # every other k_y line
    return torch.randn(B, C, H, W) * mask, mask


def test_stats_are_readable_before_any_sampling() -> None:
    """Declared in ``__init__``, so a caller never hits AttributeError."""
    diffusion, _ = _build("hard", "replace_freeze_dc")
    assert diffusion.reverse_stats["terminal_timestep_called"] is None
    assert diffusion.reverse_stats["effective_steps"] == 0


@pytest.mark.parametrize("reverse_mode", ["replace_freeze", "replace_freeze_dc"])
def test_hard_dc_skips_the_terminal_step(reverse_mode: str) -> None:
    """PLANTED: the #2067 shape. The model is never called at the R=1 rung."""
    diffusion, model = _build("hard", reverse_mode)
    measurement, mask = _measurement_and_mask()
    diffusion.sample(measurement=measurement, mask=mask, start_timestep=2)

    stats = diffusion.reverse_stats
    assert stats["skipped_steps"] >= 1, "the terminal step should have been inert"
    assert 0 not in model.seen, "the model must not have been called at t=0"
    assert stats["terminal_timestep_called"] == min(model.seen)
    assert stats["terminal_timestep_called"] != 0, (
        "terminal_timestep_called must be the lowest EXECUTED step, not the tail of the schedule"
    )


def test_soft_dc_runs_the_terminal_step() -> None:
    """The other pole: no step is inert under a non-hard DC, so t=0 runs."""
    diffusion, model = _build("soft", "replace_freeze_dc")
    measurement, mask = _measurement_and_mask()
    diffusion.sample(measurement=measurement, mask=mask, start_timestep=2)

    stats = diffusion.reverse_stats
    assert stats["skipped_steps"] == 0
    assert 0 in model.seen, "soft DC has no inert-step shortcut; t=0 must run"
    assert stats["terminal_timestep_called"] == 0


def test_effective_and_skipped_partition_the_schedule() -> None:
    diffusion, model = _build("hard", "replace_freeze_dc")
    measurement, mask = _measurement_and_mask()
    diffusion.sample(measurement=measurement, mask=mask, start_timestep=3)

    stats = diffusion.reverse_stats
    assert stats["effective_steps"] + stats["skipped_steps"] == len(stats["schedule"])
    assert stats["effective_steps"] == len(model.seen)


def test_schedule_is_published_descending_from_the_requested_head() -> None:
    """The reveal partition is defined against this schedule, so it is recorded."""
    diffusion, _ = _build("hard", "replace_freeze_dc")
    measurement, mask = _measurement_and_mask()
    diffusion.sample(measurement=measurement, mask=mask, start_timestep=3)

    schedule = diffusion.reverse_stats["schedule"]
    assert schedule[0] == 3
    assert schedule == sorted(schedule, reverse=True)
    assert len(set(schedule)) == len(schedule)


def test_wrapper_delegates_rather_than_recomputing() -> None:
    """``ColdMRISampler`` owns no schedule, so its record must be the inner one."""
    model = _Identity()
    sampler = ColdMRISampler(
        model=model,
        num_timesteps=8,
        max_acceleration=4.0,
        center_fraction=0.08,
        dc_method="hard",
        reverse_mode="replace_freeze_dc",
        kspace_log_scaled=False,
    )
    measurement, mask = _measurement_and_mask()
    sampler.sample(measurement=measurement, mask=mask, start_timestep=2)
    assert sampler.reverse_stats == sampler._diffusion.reverse_stats


def test_the_generator_stashes_the_record_from_its_per_call_sampler() -> None:
    """The real chain, not a stub: generator.sample -> sampler -> _last_reverse_stats.

    ``KSpaceColdDiffusionGenerator.sample`` builds its sampler per call and
    discards it, reading the record off with ``getattr(sampler, "reverse_stats",
    None)``. That lookup is the facade shape this whole change is about: if it
    ever returned ``None`` for a cold_mri sampler, every ``val_reverse_*`` column
    would silently read as "no record" and the helper tests would stay green.

    Also the end-to-end witness for #2067 on a real generator — a ladder with
    ``train_identity_rung`` schedules t=0 and the model is never called there.
    """
    from spectramr.models.generators.kspace_cold_diffusion_generator import (
        KSpaceColdDiffusionGenerator,
    )

    gen = KSpaceColdDiffusionGenerator(
        in_channels=2,
        out_channels=2,
        base_channels=8,
        num_res_blocks=1,
        backbone_type="complex_unet",
        attention_type="none",
        timesteps=8,
        sampling_steps=8,
        dc_method="hard",
        reverse_sampling_mode="replace_freeze_dc",
        kspace_log_scaled=False,
        force_pure_kspace=True,
        condition_with_smaps=False,
        acceleration_type="equispaced",
        base_acceleration=1.0,
        max_acceleration=4.0,
        center_fraction=0.08,
        train_identity_rung=True,
    )
    assert gen.last_reverse_stats is None, "no sampling has run yet"

    measurement, mask = _measurement_and_mask()
    gen.sample(measurement=measurement, mask=mask, inference_timesteps=8, start_timestep=2)

    stats = gen.last_reverse_stats
    assert stats is not None, "the per-call sampler's record was not stashed"
    assert stats["schedule"][-1] == 0, "t=0 IS scheduled"
    assert stats["terminal_timestep_called"] == 1, "...and the model is never called there"
    assert stats["skipped_steps"] >= 1


@pytest.mark.unit
def test_reverse_stats_carries_the_mode_that_produced_the_schedule():
    """A consumer reasoning about WHICH step wrote a coefficient needs to know
    whether steps write once and freeze.

    Reading it off the sampler object works and is what a caller reached for
    first; carrying it on the record keeps ONE transport for "what the last
    sample call did" rather than two (non-negotiable 17).
    """
    from spectramr.models.diffusion.kspace_process import VALID_REVERSE_MODES

    for mode in sorted(VALID_REVERSE_MODES):
        diffusion, _ = _build("hard", mode)
        assert diffusion.reverse_stats["reverse_mode"] == mode


# ---------------------------------------------------------------------------
# `replace_freeze_dc_t0` -- the t=0-terminal keying (#2067 step 2)
#
# The defect it addresses: under the default keying the step at `t` writes what
# the NEXT rung reveals, so on a ladder whose mask(0) is all-ones the call
# labelled t=1 writes every remaining coefficient and the scheduled t=0 step is
# inert and skipped. Measured on the reference ladder before this mode existed:
# terminal_timestep_called == 1.
# ---------------------------------------------------------------------------

import torch.nn as nn  # noqa: E402

_REFERENCE_LADDER = {
    "num_timesteps": 29,
    "max_acceleration": 32.0,
    "base_acceleration": 1.0,
    "mask_type": "variable_density",
    "seed": 42,
}


def _ladder(**over):
    """The `experiment_11_attention_none` ladder: mask(0) is all-ones."""
    return KSpaceUndersamplingProcess(
        **{**_REFERENCE_LADDER, "train_identity_rung": True, "device": "cpu", **over}
    )


class _Recorder(nn.Module):
    """Records the rung each call is made at; predicts a dense plane so an
    unwritten coefficient stays exactly zero and is countable."""

    def __init__(self, process):
        super().__init__()
        self.timesteps: list[int] = []
        self.kspace_process = process

    def forward(self, x, t):
        self.timesteps.append(int(t[0]))
        return torch.ones_like(x)


def _sample(mode, process=None, start=6):
    process = process or _ladder()
    model = _Recorder(process)
    sampler = PhysicsInformedColdDiffusion(
        model=model,
        num_timesteps=29,
        max_acceleration=32.0,
        dc_method="hard",
        reverse_mode=mode,
        sampling_steps=8,
        kspace_log_scaled=False,
    )
    x = torch.randn(1, 2, 64, 64)
    _, mask = process.q_sample(x, torch.full((1,), start, dtype=torch.long))
    out = sampler.sample(x * mask, mask, start_timestep=start)
    return model, sampler, out


@pytest.mark.unit
def test_default_keying_never_calls_the_terminal_rung():
    """The planted violation: this is the #2067 defect, still true by design."""
    model, sampler, _ = _sample("replace_freeze_dc")

    assert sampler.reverse_stats["terminal_timestep_called"] == 1
    assert 0 not in model.timesteps


@pytest.mark.unit
def test_t0_keying_calls_the_model_at_the_terminal_rung():
    """The decisive write is made by a call whose time embedding says t=0."""
    model, sampler, _ = _sample("replace_freeze_dc_t0")

    assert sampler.reverse_stats["terminal_timestep_called"] == 0
    assert model.timesteps[-1] == 0


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["replace_freeze_dc", "replace_freeze_dc_t0"])
def test_neither_keying_leaves_an_unwritten_coefficient(mode):
    """Re-keying the reveal must not open a hole: a coefficient no step writes
    stays zero forever and shows up as a low-pass blob."""
    _, _, out = _sample(mode)

    assert int((out == 0).sum()) == 0


@pytest.mark.unit
def test_t0_keying_costs_no_extra_model_calls():
    """It moves which rung is skipped, not how many steps run: the head step
    reveals nothing under the new keying, exactly as t=0 did under the old."""
    old_model, old_sampler, _ = _sample("replace_freeze_dc")
    new_model, new_sampler, _ = _sample("replace_freeze_dc_t0")

    assert len(new_model.timesteps) == len(old_model.timesteps)
    assert new_sampler.reverse_stats["effective_steps"] == (
        old_sampler.reverse_stats["effective_steps"]
    )


@pytest.mark.unit
def test_t0_keying_refuses_a_ladder_that_never_reaches_zero():
    """Without the identity rung the floor is 1, so the terminal call would land
    at t=1 and the mode would silently behave like the one it replaces."""
    process = _ladder(train_identity_rung=False)
    assert process.min_meaningful_timestep() == 1

    with pytest.raises(ValueError, match="terminal call would"):
        PhysicsInformedColdDiffusion(
            model=_Recorder(process),
            num_timesteps=29,
            max_acceleration=32.0,
            dc_method="hard",
            reverse_mode="replace_freeze_dc_t0",
            kspace_log_scaled=False,
        )


@pytest.mark.unit
def test_the_skip_gate_and_the_write_read_one_owner():
    """They were two copies of the reveal expression. A step is skipped exactly
    when the support it would write is empty -- for every mode."""
    import inspect

    from spectramr.models.diffusion import kspace_process as kp

    source = inspect.getsource(kp.PhysicsInformedColdDiffusion)
    assert source.count("def _reveal_support") == 1
    # Neither loop may re-derive the mask itself.
    assert "mask_next.float() * (1.0 - committed)" not in source
    assert inspect.getsource(kp.PhysicsInformedColdDiffusion._step_reveals_anything).count(
        "_reveal_support"
    ) == 1
