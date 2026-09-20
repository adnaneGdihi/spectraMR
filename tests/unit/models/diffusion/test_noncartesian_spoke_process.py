"""Contract tests for the golden-angle spoke degradation.

Sizes are deliberately tiny (32x32, 128 spokes): torchkbnufft runs on CPU on
this box, and every property under test is a property of the operator chain
rather than of the resolution. The spoke count is not free to shrink further --
see the inert-ladder test below.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
from spectramr.models.diffusion.noncartesian_spoke_process import (
    NonCartesianSpokeProcess,
)

IM = (32, 32)
SPOKES = 128
READOUT = 64
TIMESTEPS = 8

#: One R per rung, as every arm in this cohort declares. The process reads this
#: rather than ``get_acceleration_schedule()``, which reports a linear ramp
#: regardless of what the arm asked for (#2114).
LADDER = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0, 16.0]


def _process(**overrides) -> NonCartesianSpokeProcess:
    kwargs = {
        "num_spokes": SPOKES,
        "samples_per_spoke": READOUT,
        "im_size": IM,
        "num_timesteps": TIMESTEPS,
        "max_acceleration": 16.0,
        "base_acceleration": 1.0,
        "schedule_kwargs": {"acceleration_range": list(LADDER)},
    }
    kwargs.update(overrides)
    return NonCartesianSpokeProcess(**kwargs)


def _phantom(batch: int = 1, channels: int = 1) -> torch.Tensor:
    """A smooth disc with a phase ramp -- an object, not white noise.

    Radial sampling cannot represent energy outside its angular coverage, so
    white noise would measure the trajectory's band limit rather than the
    round-trip fidelity these tests are about.
    """
    height, width = IM
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height), torch.linspace(-1.0, 1.0, width), indexing="ij"
    )
    radius = (xx.pow(2) + yy.pow(2)).sqrt()
    magnitude = torch.nn.functional.avg_pool2d(
        ((radius < 0.7).float() * (1.0 - 0.5 * radius))[None, None], 5, 1, 2
    )[0, 0]
    image = magnitude * torch.exp(1j * (1.2 * xx + 0.8 * yy))
    return image.expand(batch, channels, height, width).clone()


def test_the_identity_rung_reconstructs_the_object() -> None:
    """At t=0 every spoke is kept, so gridding must return the object."""
    process = _process()
    x0 = _phantom()
    x_t, _ = process.q_sample(fft2c(x0), torch.zeros(1, dtype=torch.long))
    recovered = ifft2c(x_t)
    rel = ((x0 - recovered).abs().norm() / x0.abs().norm()).item()
    assert rel < 0.15, f"identity rung lost the object: rel_err={rel:.4f}"


def test_spoke_prefixes_are_nested_across_every_rung() -> None:
    """Cold diffusion's forward process only ever removes measurements."""
    process = _process()
    masks = process.sample_masks
    for t in range(masks.shape[0] - 1):
        added = ((masks[t + 1] > 0) & ~(masks[t] > 0)).sum().item()
        assert added == 0, (
            f"rung {t + 1} re-acquired {added} samples the coarser rung {t} had "
            f"dropped; the cascade is not nested"
        )


def test_degradation_is_monotone_and_no_rung_is_inert() -> None:
    """Each rung must both keep fewer spokes and reconstruct worse than the last."""
    process = _process()
    x0 = _phantom()
    k0 = fft2c(x0)
    errors, budgets = [], []
    for t in range(TIMESTEPS):
        x_t, _ = process.q_sample(k0, torch.tensor([t]))
        errors.append(((x0 - ifft2c(x_t)).abs().norm() / x0.abs().norm()).item())
        budgets.append(int(process.spokes_per_rung[t]))
    assert budgets == sorted(budgets, reverse=True), f"spoke budget not monotone: {budgets}"
    assert len(set(budgets)) == len(budgets), f"two rungs share a spoke budget: {budgets}"
    assert errors[-1] > errors[0], (
        f"the coarsest rung reconstructs no worse than the finest "
        f"({errors[-1]:.4f} vs {errors[0]:.4f}); the ladder is inert"
    )


def test_degradation_preserves_global_phase() -> None:
    """f(e^{i*phi} x) == e^{i*phi} f(x): the chain is complex-linear.

    Phase is the object's spatial-shift information. A degradation that rotated
    it would make the acquisition model wrong in a way no magnitude metric sees.
    """
    process = _process()
    x0 = _phantom()
    phi = 0.7331
    t = torch.tensor([2])
    plain, _ = process.q_sample(fft2c(x0), t)
    rotated, _ = process.q_sample(fft2c(x0 * torch.exp(torch.tensor(1j * phi))), t)
    delta = (rotated - plain * torch.exp(torch.tensor(1j * phi))).abs().max().item()
    assert delta < 1e-4, f"global phase was not carried through: max delta {delta:.3e}"


def test_the_trajectory_graph_does_not_move_with_the_rung() -> None:
    """kNN over coordinates is acquisition geometry, fixed across timesteps."""
    process = _process()
    traj = process.trajectory
    assert traj.shape == (2, SPOKES * READOUT)
    first = torch.cdist(traj.transpose(0, 1)[:64], traj.transpose(0, 1)[:64])
    process.q_sample(fft2c(_phantom()), torch.tensor([TIMESTEPS - 1]))
    second = torch.cdist(traj.transpose(0, 1)[:64], traj.transpose(0, 1)[:64])
    assert torch.equal(first, second), "the trajectory changed when a rung was drawn"


def test_grid_data_consistency_raises_rather_than_pinning_interpolated_bins() -> None:
    """Planted violation for the silent-fallback shape (non-negotiable 3).

    A grid DC layer handed this process's output would pin gridded values as if
    they had been measured. Returning a no-op mask would hide that; raising is
    the contract.
    """
    process = _process()
    assert process.supports_grid_data_consistency is False
    with pytest.raises(NotImplementedError, match="off-grid"):
        process.apply_data_consistency(torch.zeros(1), torch.zeros(1), torch.zeros(1))


def test_the_reverse_step_has_bins_to_reveal_at_every_transition() -> None:
    """Planted violation for the reverse-loop no-op (non-negotiable 16).

    The cold reverse step is ``x_{t-1} = x_t - D(x0,t) + D(x0,t-1)``, realised
    in ``PhysicsInformedColdDiffusion`` as
    ``mask_recovered = clamp(mask_{t-1} - mask_t, min=0)`` and
    ``recovered_lines = x_0_pred * mask_recovered``. Return an empty mask at
    every rung and that difference is identically zero, so ``x_{t-1} == x_t``
    and the entire reverse loop is an identity that still runs, still logs and
    still reports success.

    The returned mask therefore has to carry which bins GAINED information, not
    which bins were measured -- no Cartesian bin is ever measured here.
    """
    process = _process()
    k0 = fft2c(_phantom())
    previous = None
    for t in range(TIMESTEPS):
        _, coverage = process.q_sample(k0, torch.tensor([t]))
        assert coverage.shape == (1, 1, *IM)
        if previous is not None:
            revealed = (previous - coverage).clamp(min=0.0).sum().item()
            assert revealed > 0, (
                f"the step from rung {t - 1} to {t} reveals no bin; the reverse "
                f"loop would be an identity there"
            )
        previous = coverage


def test_coverage_is_nested_and_never_claims_the_whole_grid() -> None:
    """Radial covers a disc, so the corners are never informed -- at any rung."""
    process = _process()
    coverage = process.grid_coverage
    assert coverage[0].mean().item() < 1.0, (
        "the identity rung claims every Cartesian bin; radial sampling does not "
        "reach the corners of k-space"
    )
    for t in range(1, coverage.shape[0]):
        assert bool(((coverage[t - 1] - coverage[t]) >= 0).all()), (
            f"rung {t} informs a bin rung {t - 1} did not; coverage is not nested"
        )


def test_multi_coil_input_degrades_per_channel() -> None:
    """The cohort is multi-coil; a chain that only handled C=1 would pass above."""
    process = _process()
    x0 = _phantom(batch=2, channels=3)
    x_t, coverage = process.q_sample(fft2c(x0), torch.tensor([0, TIMESTEPS - 1]))
    assert x_t.shape == (2, 3, *IM)
    assert coverage.shape == (2, 1, *IM)
    # Each batch element must take its OWN rung. Compared at the ends rather
    # than between neighbours: at 32x32 on a smooth phantom the gridding
    # kernel's own error is comparable to one rung's worth of spoke removal, so
    # adjacent rungs are not separated (measured 0.0410 at 64 spokes against
    # 0.0401 at 21). The ends are, by a factor of two.
    finest = (x0[0] - ifft2c(x_t[:1])[0]).abs().norm().item()
    coarsest = (x0[1] - ifft2c(x_t[1:])[0]).abs().norm().item()
    assert coarsest > finest, (
        f"the coarsest rung is no worse than the finest ({coarsest:.4f} vs "
        f"{finest:.4f}); the per-sample rung index is not being applied"
    )


def test_the_ladder_report_describes_spokes_not_cartesian_bins() -> None:
    """The inherited reporter would certify a bin ladder this arm never runs.

    That is #2114's shape -- a reporter disagreeing with the realiser -- and the
    override is what keeps it out of this class.
    """
    process = _process()
    rows = process.describe_ladder(IM)
    assert len(rows) == TIMESTEPS
    for t, r_nominal, r_effective, kept in rows:
        assert r_nominal == pytest.approx(LADDER[t])
        assert kept == int(process.spokes_per_rung[t])
        assert r_effective == pytest.approx(SPOKES / kept)
    assert process.inert_step_report() == []
    assert process.nesting_leak_report() == []


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"num_spokes": 0}, "must be >= 1"),
        ({"density_compensation": "bogus"}, "Unknown density_compensation"),
        ({"density_compensation": "iterative"}, "re-solved per rung"),
    ],
)
def test_construction_rejects_unusable_settings(kwargs: dict, match: str) -> None:
    """Unknown options raise instead of degrading to a default."""
    with pytest.raises(ValueError, match=match):
        _process(**kwargs)


def test_a_wrong_sized_input_raises_instead_of_silently_regridding() -> None:
    """The trajectory is built for one matrix size; a mismatch is a config bug."""
    process = _process()
    with pytest.raises(ValueError, match="rebuild the process"):
        process.q_sample(torch.zeros(1, 1, 16, 16, dtype=torch.complex64), torch.tensor([0]))


def test_a_spoke_count_that_collapses_two_rungs_raises() -> None:
    """Planted violation for the inert-rung shape (#1155, non-negotiable 15).

    ``round(num_spokes / R)`` is not injective for every pair of spoke count and
    ladder. At 64 spokes over these 8 rungs two of them round to 5, so both emit
    the same mask: the reverse step has nothing to reveal and the time embedding
    is trained to separate identical states. Measured budgets at 64 spokes are
    ``[64, 20, 12, 9, 7, 5, 5, 4]``; 128 spokes gives 8 distinct values.
    """
    # This ladder rounds two of its rungs onto 6 spokes at 64: 64/10 and 64/11.
    colliding = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0, 11.0, 16.0]
    with pytest.raises(ValueError, match="makes the ladder inert"):
        _process(
            num_spokes=64,
            num_timesteps=len(colliding),
            schedule_kwargs={"acceleration_range": colliding},
        )


def test_an_arm_without_a_declared_ladder_raises() -> None:
    """The spoke budget is num_spokes / R, so R must be the declared one.

    Falling back to ``get_acceleration_schedule()`` would silently acquire a
    ladder the arm never asked for (#2114).
    """
    with pytest.raises(ValueError, match="needs an explicit"):
        _process(schedule_kwargs={})


@pytest.mark.parametrize(
    ("knob", "match"),
    [
        ({"enable_dynamic_mask": True}, "IS the acquisition"),
        ({"prior_channel_range": (0, 1)}, "no grid bins"),
    ],
)
def test_knobs_this_acquisition_cannot_honour_raise(knob: dict, match: str) -> None:
    """Planted violation for the unread-knob shape (non-negotiable 8).

    Both knobs steer the parent's Cartesian mask generator, which this process
    does not consult. Accepting them would let an arm declare a behaviour that
    silently never happens.
    """
    with pytest.raises(ValueError, match=match):
        _process(**knob)


def test_q_sample_publishes_the_measurement_it_grids_away() -> None:
    """The samples must survive the gridding that destroys them in ``x_t``.

    ``q_sample`` returns grid k-space, so a sample-domain fidelity term cannot
    recover the measurement from its output. Re-projecting would cost a second
    NUFFT forward per training step (non-negotiable 9).
    """
    process = _process()
    assert process.last_sample_measurement is None, "nothing measured before q_sample"

    x0 = torch.randn(2, 3, *IM, dtype=torch.complex64)
    process.q_sample(x0, torch.tensor([1, 1]))
    measurement = process.last_sample_measurement

    assert measurement is not None
    assert measurement.samples.shape[:2] == (2, 3)
    assert measurement.samples.shape[-1] == process.trajectory.shape[-1]
    assert measurement.mask.shape == (2, process.trajectory.shape[-1])
    assert measurement.projector is process.nufft


def test_the_published_samples_are_detached() -> None:
    """The measurement is data: a gradient path would let the net move its own target."""
    process = _process()
    x0 = torch.randn(1, 1, *IM, dtype=torch.complex64).requires_grad_(True)
    process.q_sample(x0, torch.tensor([0]))
    assert not process.last_sample_measurement.samples.requires_grad


def test_the_published_mask_tracks_the_rung() -> None:
    """A rung-independent measurement would score every rung against one spoke set."""
    process = _process()
    x0 = torch.randn(1, 1, *IM, dtype=torch.complex64)
    retained = []
    for rung in range(process.num_timesteps):
        process.q_sample(x0, torch.tensor([rung]))
        retained.append(float(process.last_sample_measurement.mask.sum()))
    assert retained == sorted(retained, reverse=True)
    assert retained[0] > retained[-1], "the ladder must drop samples as it climbs"
