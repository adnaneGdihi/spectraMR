"""Regression suite for ``RandomCartesianKSpaceAccelerator``'s ACS priority (#581).

The defect: a FLAT ACS boost (``scores[box] = 2.0``) made every ACS bin tie
exactly, so ``argsort`` fell back to flat (row-major) index order and filled the
box top-row-first. Once the budget dropped below the ACS size (R > 1/cf) the
top-K stopped partway down the box, leaving contiguous horizontal STRIPES with
an UNSAMPLED DC bin. On the exp_11 ladder at R=32x that retained 0.48% of the
target's k-space energy and its point-spread function was the ringing that put
every R=32x case at 12-20 dB.

These are invariants of the accelerator, parametrised over the whole exp_11
ladder rather than asserted at one rung.
"""

import itertools
import math
from typing import ClassVar

import pytest
import torch

from spectramr.infrastructure.physics.sampling import (
    MaskGenerator,
    RandomCartesianKSpaceAccelerator,
    _accelerator_kwarg_vocabulary,
    create_kspace_accelerator,
)

SHAPE = (1, 256, 256)


def _accelerator(**overrides):
    kwargs = {
        "num_timesteps": 28,
        "max_acceleration": 32.0,
        "center_fraction": 0.08,
        "min_center_fraction": 0.02,
        "seed": 42,
    }
    kwargs.update(overrides)
    return RandomCartesianKSpaceAccelerator(**kwargs)


def _mask_at(accel, timestep):
    return accel.get_acceleration_mask(SHAPE, timestep)[0]


@pytest.mark.parametrize("timestep", range(0, 28, 3))
def test_dc_bin_is_always_sampled(timestep):
    """DC is the unique score maximum, so it survives every budget.

    Pre-fix, the flat boost let ``argsort`` break ties by flat index and DC went
    unsampled in 4 of 6 shape/seed combinations once R > 12.5.
    """
    mask = _mask_at(_accelerator(), timestep)
    assert mask[128, 128], f"DC bin unsampled at t={timestep}"


def test_nesting_holds_across_every_timestep():
    """M_{t+1} subset-of M_t — the invariant cold diffusion's forward pass assumes.

    Graded scores must stay timestep-INVARIANT; only the budget may change.
    """
    accel = _accelerator()
    previous = None
    for timestep in range(28):
        mask = _mask_at(accel, timestep)
        if previous is not None:
            added = int((mask & ~previous).sum())
            assert added == 0, f"t={timestep} ADDED {added} bins (nesting broken)"
        previous = mask


def _core_radius(accel) -> float:
    """Radius of the disc the accelerator guarantees at every rung."""
    total = SHAPE[1] * SHAPE[2]
    return math.sqrt(total * accel._guaranteed_core_fraction(accel.center_fraction) / math.pi)


@pytest.mark.parametrize("timestep", [14, 17, 21, 25, 27])
def test_high_acceleration_support_is_centred_but_is_not_a_disc(timestep):
    """Past the ACS-budget crossover: a solid centred core AND high frequencies.

    Two opposite defects meet at these timesteps and this test has to exclude
    both, which is why it does not simply assert a shape.

    #581, the original: a FLAT ACS boost made every ACS bin tie exactly, so
    ``argsort`` fell back to row-major order and filled the box top-row-first,
    leaving horizontal STRIPES with DC itself unsampled.

    #1069, the over-correction: grading that boost killed the stripes, but it
    also put the entire ACS band ([1.5, 2.0]) strictly above every peripheral
    score ([0, 1)). Once the budget fell below the nominal ACS the top-K never
    reached a random bin, so the mask became a pure low-pass DISC acquiring
    nothing beyond 20% of the Nyquist radius. An earlier revision of this test
    asserted exactly that -- ``fill > 0.95`` of the bounding circle -- and so
    locked the collapse in as the expected behaviour.

    The invariant that excludes both at once: the guaranteed core is SOLID
    (no stripes) and the support extends well beyond it (no disc).
    """
    accel = _accelerator()
    mask = _mask_at(accel, timestep)
    rows, cols = torch.nonzero(mask, as_tuple=True)

    assert abs(float(rows.float().mean()) - 128.0) < 2.0, "support centroid off-centre in ky"
    assert abs(float(cols.float().mean()) - 128.0) < 2.0, "support centroid off-centre in kx"

    grid = torch.arange(256, dtype=torch.float32) - 128
    all_radii = torch.sqrt(grid[:, None] ** 2 + grid[None, :] ** 2)

    # Anti-#581: every bin of the guaranteed core is sampled. Strictly stronger
    # than the old bounding-circle fill ratio, which a stripe pattern could in
    # principle satisfy at a small enough enclosing radius.
    core = all_radii <= _core_radius(accel)
    assert bool(mask[core].all()), (
        f"the guaranteed core is not solid at t={timestep} — stripe pattern"
    )

    # Anti-#1069: data was actually ACQUIRED beyond half-Nyquist. This is the
    # decisive quantity; peak-to-sidelobe ratio is not, because a disc's
    # sidelobes are low-amplitude yet perfectly coherent and it scores *better*
    # than an incoherent mask at high acceleration.
    far = all_radii > 0.5 * float(all_radii.max())
    assert int((mask & far).sum()) > 0, (
        f"nothing sampled beyond half-Nyquist at t={timestep} — the mask is a "
        "low-pass disc, so the task is extrapolation rather than compressed sensing"
    )


def test_support_grows_monotonically_outward_from_dc():
    """Sampling extends outward: the max radius never shrinks as the budget grows."""
    accel = _accelerator()
    previous_radius = None
    for timestep in range(27, -1, -1):  # most accelerated -> least
        mask = _mask_at(accel, timestep)
        rows, cols = torch.nonzero(mask, as_tuple=True)
        radius = float(torch.sqrt((rows.float() - 128) ** 2 + (cols.float() - 128) ** 2).max())
        if previous_radius is not None:
            assert radius >= previous_radius - 1e-6, (
                f"support radius shrank at t={timestep} as the budget grew"
            )
        previous_radius = radius


def test_the_guaranteed_core_is_exhausted_before_any_peripheral_bin():
    """Every CORE bin outranks every random bin, so the core fills first.

    The core, not the nominal ACS. Those coincide only when no floor is
    declared; when one is, ``min_center_fraction`` sets the core and the band
    between it and ``center_fraction`` is left to compete with the periphery on
    equal terms. Asserting the *nominal* band here is what #1069 measured as the
    disc collapse.
    """
    accel = _accelerator()
    core_radius = _core_radius(accel)
    assert core_radius < math.sqrt(SHAPE[1] * SHAPE[2] * accel.center_fraction / math.pi), (
        "fixture must declare a floor strictly below center_fraction for this to bite"
    )
    for timestep in range(28):
        mask = _mask_at(accel, timestep)
        grid = torch.arange(256, dtype=torch.float32) - 128
        core = torch.sqrt(grid[:, None] ** 2 + grid[None, :] ** 2) <= core_radius
        assert bool(mask[core].all()), (
            f"a peripheral bin was taken at t={timestep} while core bins were unsampled"
        )


def test_a_declared_floor_restores_high_frequency_coverage():
    """Regression for #1069: the floor is what buys incoherent aliasing.

    Sizing the guaranteed core from ``center_fraction`` (0.08) against the R=32
    budget (1/32 = 3.125%) is unsatisfiable arithmetic -- the core alone is 2.6x
    the budget -- so the realised mask was 100% core and acquired nothing beyond
    14% of the Nyquist radius. Sizing it from ``min_center_fraction`` (0.02)
    leaves 1.125% of k-space for genuinely random bins at the top rung.
    """
    grid = torch.arange(256, dtype=torch.float32) - 128
    all_radii = torch.sqrt(grid[:, None] ** 2 + grid[None, :] ** 2)
    far = all_radii > 0.5 * float(all_radii.max())

    without = _mask_at(_accelerator(min_center_fraction=None), 27)
    with_floor = _mask_at(_accelerator(min_center_fraction=0.02), 27)

    assert int((without & far).sum()) == 0, (
        "fixture drifted: without a floor the top rung should acquire NO "
        "high-frequency data, which is the defect this guards"
    )
    assert int((with_floor & far).sum()) > 0
    # Same budget either way -- the floor redistributes the samples, it does not
    # buy more of them.
    assert int(without.sum()) == int(with_floor.sum())


def test_a_declared_floor_does_not_break_nesting():
    """The redistribution must not cost the cold-diffusion forward invariant."""
    accel = _accelerator(min_center_fraction=0.02)
    previous = None
    for timestep in range(28):
        mask = _mask_at(accel, timestep)
        if previous is not None:
            added = int((mask & ~previous).sum())
            assert added == 0, f"t={timestep} ADDED {added} bins (nesting broken)"
        previous = mask


def test_no_declared_floor_leaves_the_core_at_center_fraction():
    """The fallback keeps un-migrated arms byte-identical (open/closed)."""
    accel = _accelerator(min_center_fraction=None)
    assert accel._guaranteed_core_fraction(0.08) == pytest.approx(0.08)


def test_a_floor_above_center_fraction_is_clamped_not_honoured():
    """``_guaranteed_core_fraction`` never returns a core wider than the nominal ACS."""
    accel = _accelerator(min_center_fraction=None)
    accel.min_center_fraction = 0.5
    assert accel._guaranteed_core_fraction(0.08) == pytest.approx(0.08)


def test_min_center_fraction_is_accepted_not_swallowed():
    """The knob is read, not dropped into ``**kwargs`` (pitfall #15)."""
    accel = _accelerator(min_center_fraction=0.02)
    assert accel.min_center_fraction == pytest.approx(0.02)


def test_min_center_fraction_defaults_to_center_fraction():
    accel = RandomCartesianKSpaceAccelerator(num_timesteps=1000, center_fraction=0.08, max_acceleration=8.0)
    assert accel.min_center_fraction == pytest.approx(0.08)


def test_unsatisfiable_acs_floor_raises_at_build():
    """An EXPLICIT ACS floor wider than the tightest budget is a config error.

    At max_acceleration=32 the budget is 3.125% of k-space; a 6% ACS floor
    cannot fit, so the declared floor is unreachable. Fail loud (#9).
    """
    with pytest.raises(ValueError, match="exceeds the sampling budget"):
        RandomCartesianKSpaceAccelerator(num_timesteps=1000,
            max_acceleration=32.0, center_fraction=0.10, min_center_fraction=0.06
        )


@pytest.mark.parametrize("max_acceleration", [32.0, 64.0, 128.0])
def test_default_acs_never_raises_however_tight_the_budget(max_acceleration):
    """A budget narrower than the nominal ACS is the normal high-R regime.

    With centre-out grading the realised ACS is ``min(budget, center_fraction)``
    and shrinks with acceleration by construction, so an UNdeclared floor must
    never raise. The library default ``center_fraction=0.0325`` exceeds the
    budget at any ``max_acceleration`` above ~30 — an unconditional budget check
    rejected the default everywhere, which is how this was caught
    (``test_acceleration_levels``, ``test_sampling_expansion``).
    """
    accel = RandomCartesianKSpaceAccelerator(num_timesteps=1000, max_acceleration=max_acceleration)
    mask = accel.get_acceleration_mask(SHAPE, accel.num_timesteps - 1)[0]
    assert mask[128, 128], "DC unsampled at the tightest budget"
    assert int(mask.sum()) > 0


def test_inverted_acs_bounds_raise():
    with pytest.raises(ValueError, match="exceeds center_fraction"):
        RandomCartesianKSpaceAccelerator(num_timesteps=1000,
            max_acceleration=4.0, center_fraction=0.02, min_center_fraction=0.08
        )


@pytest.mark.parametrize("timestep", [0, 7, 14, 17, 21, 25, 27])
def test_ladder_retains_the_low_frequency_energy(timestep):
    """Every rung must keep the bulk of a concentrated radial spectrum.

    Pre-fix the R=32x rung kept 0.48% of the real target's k-space energy,
    because its stripes sat off-centre and missed DC entirely. The stand-in
    spectrum here is radially decaying (energy concentrated near DC, as MRI
    k-space is); the guard tracks "the mask is centred on DC", not any one
    dataset's exact spectrum.
    """
    grid = torch.arange(256, dtype=torch.float32) - 128
    radius = torch.sqrt(grid[:, None] ** 2 + grid[None, :] ** 2)
    energy = 1.0 / (1.0 + radius) ** 3

    mask = _mask_at(_accelerator(), timestep)
    retained = float((energy * mask).sum() / energy.sum())
    assert retained > 0.90, f"t={timestep} retained only {retained:.2%} of spectral energy"


# ---------------------------------------------------------------------------
# Shrinking ACS for the variable-density families (#534)
#
# ``VDCartesian1DAccelerator`` and ``VDCartesian2DGaussianAccelerator`` held the
# ACS at its nominal width for the whole ladder while accepting
# ``min_center_fraction`` through ``**kwargs`` and never reading it (pitfall
# #15). An always-sampled 8% band IS the entire budget past R=12.5, so the
# exp_11 arms' declared 16x and 32x rungs realised the same ~12x mask -- nine
# entries in ``scripts/ci/ladder_baseline.txt``.
#
# Both classes now take a top-K of ONE fixed ranking (ACS first, centre-out;
# then the family's weighted permutation) rather than painting the band and
# topping up. Paint-then-top-up makes the peripheral count grow with t once the
# band shrinks faster than the budget, which re-admits samples the cold
# diffusion forward process cannot remove again.
# ---------------------------------------------------------------------------

LADDER = [2.0, 4.0, 8.0, 10.0, 12.0, 16.0, 32.0]
SHRINK_FAMILIES = ["variable_density_1d", "variable_density_2d_gaussian"]


def _cascade(pattern: str, *, min_center_fraction=None, size: int = 256, budget=True):
    """The 28 masks of one exp_11 cascade, as float tensors ``[H, W]``."""
    from spectramr.infrastructure.physics.sampling import create_kspace_accelerator

    kwargs = {
        "num_timesteps": 28,
        "max_acceleration": 32.0,
        "base_acceleration": 2.0,
        "center_fraction": 0.08,
        "seed": 42,
        "acceleration_schedule": "step",
        "acceleration_range": LADDER,
    }
    if min_center_fraction is not None:
        kwargs["min_center_fraction"] = min_center_fraction
    accel = create_kspace_accelerator(pattern, **kwargs)
    inner = getattr(accel, "accelerator", accel)
    inner.enforces_sample_budget = budget
    return [inner.get_acceleration_mask((1, size, size), t).float()[0] for t in range(28)]


def _realised_accelerations(masks):
    return [1.0 / max(float(m.mean()), 1e-9) for m in masks]


@pytest.mark.unit
@pytest.mark.parametrize("pattern", SHRINK_FAMILIES)
def test_no_declared_floor_leaves_the_cascade_byte_identical(pattern):
    """The blast-radius guarantee: no floor below the ACS => nothing moves.

    Every arm in the corpus is in this state, so this is what makes the change
    safe for ``active/`` and ``validated/``. A floor equal to ``center_fraction``
    (0.08) no longer fits the 32x budget and is refused at build (#2061).
    """
    baseline = _cascade(pattern)
    shifted = _cascade(pattern, min_center_fraction=None)
    for before, after in zip(baseline, shifted, strict=True):
        assert torch.equal(before, after)


@pytest.mark.unit
@pytest.mark.parametrize("pattern", SHRINK_FAMILIES)
def test_a_declared_floor_realises_the_declared_ladder(pattern):
    """The payoff: the 32x rung is a 32x mask, not a relabelled 12x one."""
    clamped = max(_realised_accelerations(_cascade(pattern)))
    assert clamped < 13.0, "precondition: the nominal ACS clamps this family"

    realised = _realised_accelerations(_cascade(pattern, min_center_fraction=0.02))
    assert max(realised) >= 31.0
    # The two rungs that used to collapse onto ~12x are now distinct.
    assert len({round(r) for r in realised if r > 13.0}) >= 2


@pytest.mark.unit
@pytest.mark.parametrize("pattern", SHRINK_FAMILIES)
def test_a_shrinking_acs_stays_nested(pattern):
    """``M_{t+1}`` subset-of ``M_t`` -- what cold diffusion's forward pass assumes.

    Measured with the budget top-up disabled, which is a separate pre-existing
    source of re-added samples in the 1D family (512 of them on the unmodified
    cascade) and is not what this change is about.
    """
    masks = _cascade(pattern, min_center_fraction=0.02, budget=False)
    added = sum(int(((masks[i + 1] > 0) & (masks[i] == 0)).sum()) for i in range(len(masks) - 1))
    assert added == 0, f"{added} samples re-entered the mask as acceleration rose"


@pytest.mark.unit
@pytest.mark.parametrize("pattern", SHRINK_FAMILIES)
def test_the_shrinking_band_keeps_dc(pattern):
    """Truncating the ACS centre-out, not from one end, is why nesting holds.

    A raster-order prefix would drop DC at the tight rungs -- the #581 failure
    mode in a different family.
    """
    masks = _cascade(pattern, min_center_fraction=0.02, size=64, budget=False)
    for t, m in enumerate(masks):
        assert m[32, 32] > 0, f"DC unsampled at t={t}"


@pytest.mark.unit
def test_the_helper_is_a_no_op_without_a_floor():
    """``_current_center_fraction`` is the single owner of the shrink."""
    from spectramr.infrastructure.physics.sampling import VDCartesian1DAccelerator

    accel = VDCartesian1DAccelerator(num_timesteps=28, max_acceleration=32.0, center_fraction=0.08)
    assert accel.min_center_fraction is None
    assert all(accel._current_center_fraction(t, 0.08) == 0.08 for t in range(28))

    accel = VDCartesian1DAccelerator(
        num_timesteps=28, max_acceleration=32.0, center_fraction=0.08, min_center_fraction=0.02
    )
    fractions = [accel._current_center_fraction(t, 0.08) for t in range(28)]
    assert fractions[0] == pytest.approx(0.08)
    assert fractions[-1] == pytest.approx(0.02)
    # Monotone non-increasing: the band may only close as acceleration rises.
    assert all(b <= a + 1e-12 for a, b in itertools.pairwise(fractions))


# ---------------------------------------------------------------------------
# DensityNestedKSpaceAccelerator (#1066)
#
# The deterministic-ranking families nest, but they nest by collapsing: the
# top-K of a monotone-in-radius score IS a centred disc. Measured on this same
# exp_11 ladder, ``random_cartesian`` acquires NOTHING beyond 20% of the Nyquist
# radius at R=16 and nothing beyond 14% at R=32 -- the network is asked to
# hallucinate resolution that was never sampled. These tests pin the property
# the new family adds (a weighted-random draw, so high frequencies keep being
# acquired) WITHOUT giving up the property it inherits (nesting).
# ---------------------------------------------------------------------------


def _density_nested(**overrides):
    from spectramr.infrastructure.physics.sampling import DensityNestedKSpaceAccelerator

    kwargs = {
        "num_timesteps": 28,
        "max_acceleration": 32.0,
        "base_acceleration": 2.0,
        "center_fraction": 0.08,
        "min_center_fraction": 0.02,
        "density_power": 1.6,
        "seed": 42,
        "acceleration_schedule": "step",
        "acceleration_range": LADDER,
    }
    kwargs.update(overrides)
    return DensityNestedKSpaceAccelerator(**kwargs)


def _radius(size: int = 256) -> torch.Tensor:
    axis = torch.arange(size, dtype=torch.float32) - size // 2
    return torch.sqrt(axis[:, None] ** 2 + axis[None, :] ** 2)


@pytest.mark.parametrize("line_axis", [None, "y", "x"])
def test_cascade_is_strictly_nested(line_axis):
    """The whole point: prefixes of one permutation can only ever shrink."""
    accel = _density_nested(line_axis=line_axis)
    masks = [accel.get_acceleration_mask(SHAPE, t)[0] for t in range(28)]
    added = sum(int((b & ~a).sum()) for a, b in itertools.pairwise(masks))
    assert added == 0, f"{added} bins re-entered the mask across the cascade"


def test_high_frequencies_are_acquired_at_max_acceleration():
    """The defect this family exists to fix.

    ``random_cartesian`` samples 0.00% of the bins beyond half-Nyquist at both
    R=16 and R=32 on this ladder. A weighted-random draw must not.
    """
    accel = _density_nested()
    mask = accel.get_acceleration_mask(SHAPE, 27)[0]
    radius = _radius()
    far = radius > 0.5 * float(radius.max())
    assert int((mask & far).sum()) > 0, "no high-frequency bin sampled at R=32"
    # And the outermost sample must reach most of the way to the corner, not
    # stop at a disc rim.
    assert float(radius[mask].max()) > 0.8 * float(radius.max())


def test_not_a_low_pass_disc():
    """Distinguishes a weighted DRAW from a sort of the density itself.

    Jaccard against the ideal disc of equal cardinality is ~0.995 for the
    deterministic families; a genuine draw sits far below that.
    """
    accel = _density_nested()
    mask = accel.get_acceleration_mask(SHAPE, 27)[0]
    n = int(mask.sum())
    radius = _radius()
    disc = torch.zeros(256 * 256, dtype=torch.bool)
    disc[torch.argsort(radius.flatten())[:n]] = True
    disc = disc.view(256, 256)
    jaccard = float((mask & disc).sum()) / float((mask | disc).sum())
    assert jaccard < 0.75, f"realised mask is a low-pass disc (Jaccard {jaccard:.3f})"


def test_dc_and_contiguous_core_survive_every_rung():
    """ESPIRiT needs a contiguous ACS with DC at every rung, not just at t=0."""
    accel = _density_nested()
    radius = _radius()
    # The guaranteed core is the min_center_fraction disc.
    core = radius <= math.sqrt(256 * 256 * 0.02 / math.pi)
    for t in range(28):
        mask = accel.get_acceleration_mask(SHAPE, t)[0]
        assert bool(mask[128, 128]), f"DC unsampled at t={t}"
        assert bool(mask[core].all()), f"core ACS incomplete at t={t}"


def test_density_is_monotone_in_radius():
    """Sampling rate must fall with |k| -- otherwise it is not variable density."""
    accel = _density_nested()
    mask = accel.get_acceleration_mask(SHAPE, 20)[0]
    radius = _radius()
    rmax = float(radius.max())
    rates = []
    for i in range(2, 8):  # skip the core shells, which are saturated at 1.0
        shell = (radius >= rmax * i / 8) & (radius < rmax * (i + 1) / 8)
        rates.append(float(mask[shell].float().mean()))
    assert all(b <= a + 1e-9 for a, b in itertools.pairwise(rates)), rates


def test_budget_is_exact_without_the_top_up():
    """Exact-K by construction, so ``_ensure_sample_budget`` stays uninvoked.

    That top-up ADDS bins from a second, independent permutation -- one of the
    two measured mechanisms that break nesting in the line-based families.
    """
    accel = _density_nested()
    assert accel.enforces_sample_budget is False
    for t in (0, 12, 27):
        mask = accel.get_acceleration_mask(SHAPE, t)[0]
        expected = max(1, round(256 * 256 * accel._target_sampling_fraction(t)))
        assert int(mask.sum()) == expected


def test_line_mode_samples_whole_phase_encode_lines():
    """2D Cartesian MRI can only skip entire phase-encode lines."""
    accel = _density_nested(line_axis="y")
    mask = accel.get_acceleration_mask(SHAPE, 20)[0]
    per_line = mask.float().sum(dim=1)
    assert set(per_line.unique().tolist()) <= {0.0, 256.0}


def test_ranking_is_timestep_invariant_and_seed_determined():
    a = _density_nested().get_acceleration_mask(SHAPE, 27)[0]
    b = _density_nested().get_acceleration_mask(SHAPE, 27)[0]
    c = _density_nested(seed=7).get_acceleration_mask(SHAPE, 27)[0]
    assert torch.equal(a, b), "same seed must reproduce the mask exactly"
    assert not torch.equal(a, c), "different seeds must draw different masks"


def test_core_fraction_falls_back_to_center_fraction():
    """``min_center_fraction`` is the floor; absent it, the nominal ACS is."""
    assert _density_nested(min_center_fraction=None).core_fraction == pytest.approx(0.08)
    assert _density_nested(min_center_fraction=0.02).core_fraction == pytest.approx(0.02)


def test_zero_density_power_is_a_uniform_draw():
    """p=0 makes every non-core weight equal -- the incoherent extreme."""
    accel = _density_nested(density_power=0.0)
    mask = accel.get_acceleration_mask(SHAPE, 20)[0]
    radius = _radius()
    rmax = float(radius.max())
    outer = (radius >= rmax * 0.5) & (radius < rmax * 0.75)
    far = radius >= rmax * 0.75
    assert float(mask[outer].float().mean()) == pytest.approx(
        float(mask[far].float().mean()), abs=0.05
    )


def test_equispaced_line_selection_reads_no_index_back_to_the_host(monkeypatch) -> None:
    """#2573: one ``Tensor.__index__`` per selected line is one sync each on CUDA.

    Planted: the pre-fix list comprehension indexes a Python list with each
    element of ``perm`` and trips this guard on the first selected line.
    """
    from spectramr.infrastructure.physics.sampling import UniformCartesianKSpaceAccelerator

    accelerator = UniformCartesianKSpaceAccelerator(
        num_timesteps=8, max_acceleration=8.0, base_acceleration=2.0, seed=3
    )
    expected = accelerator.get_acceleration_mask((1, 32, 32), 4)

    def refuse(self: torch.Tensor) -> int:
        raise AssertionError("a tensor element was read back as a Python index")

    monkeypatch.setattr(torch.Tensor, "__index__", refuse)
    mask = accelerator.get_acceleration_mask((1, 32, 32), 4)
    monkeypatch.undo()
    assert torch.equal(mask, expected)

@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"line_axis": "z"}, "line_axis"),
        ({"density_power": -1.0}, "density_power"),
    ],
)
def test_invalid_arguments_raise(kwargs, match):
    """No silent fallback to a default (non-negotiable #3)."""
    with pytest.raises(ValueError, match=match):
        _density_nested(**kwargs)


def test_registered_and_reachable_from_yaml():
    from spectramr.infrastructure.physics.sampling import (
        SUPPORTED_ACCELERATION_TYPES,
        create_kspace_accelerator,
    )
    from spectramr.infrastructure.physics.sampling_registry import SamplingPatternRegistry

    assert "density_nested" in SUPPORTED_ACCELERATION_TYPES
    assert SamplingPatternRegistry.resolve("density_nested") == "density_nested"
    accel = create_kspace_accelerator("density_nested", num_timesteps=28, seed=42)
    assert accel.get_acceleration_mask(SHAPE, 0).shape == SHAPE


def test_seed_mutation_on_a_reused_instance_redraws():
    """Dynamic-mask training mutates ``.seed`` on ONE reused accelerator.

    ``kspace_process.py::_generate_batch_masks_dynamic`` calls
    ``_get_accelerator`` once and then sets ``inner.seed`` per sample. A ranking
    cache keyed on shape alone would hand every sample the first draw, making
    ``enable_dynamic_mask: true`` a silent no-op. Distinct from
    ``test_ranking_is_timestep_invariant_and_seed_determined``, which builds a
    fresh instance per seed and so cannot observe this.
    """
    accel = _density_nested()
    first = accel.get_acceleration_mask(SHAPE, 20)[0].clone()
    accel.seed = 12345
    second = accel.get_acceleration_mask(SHAPE, 20)[0].clone()
    assert not torch.equal(first, second), "mutating .seed did not redraw the ranking"
    # ...and restoring the seed must restore the original draw exactly, which is
    # how the training loop leaves the accelerator for validation.
    accel.seed = 42
    assert torch.equal(accel.get_acceleration_mask(SHAPE, 20)[0], first)


class TestUnknownKwargsAreRejected:
    """``create_kspace_accelerator`` must not silently discard a construction kwarg (#1059).

    Every registered family declares ``**kwargs``, so Python rejects nothing on its own.
    ``mask_seed`` -- the spelling every experiment YAML uses -- was therefore absorbed,
    leaving ``self.seed = None`` and mask generation on the GLOBAL RNG. Still
    reproducible under ``seed_everything``, and still wrong: each call draws a fresh
    permutation instead of truncating one fixed ranking, so the cascade stops being
    nested, which is the single property cold diffusion's forward process assumes.
    """

    BASE: ClassVar[dict] = {
        "num_timesteps": 28,
        "max_acceleration": 32.0,
        "base_acceleration": 2.0,
        "center_fraction": 0.08,
        "acceleration_schedule": "step",
        "acceleration_range": [2.0, 4.0, 8.0, 10.0, 12.0, 16.0, 32.0],
    }

    def _make(self, **extra):
        return create_kspace_accelerator(acceleration_type="random_cartesian", **self.BASE, **extra)

    def test_mask_seed_raises_and_suggests_seed(self) -> None:
        """The exact miss from #1059, with the near-miss hint that shortens the next one."""
        with pytest.raises(TypeError, match="mask_seed") as excinfo:
            self._make(mask_seed=42)
        assert "did you mean 'seed'" in str(excinfo.value)

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(TypeError, match="not read by any registered"):
            self._make(no_such_knob=1)

    def test_correct_spelling_is_honoured(self) -> None:
        """The rejection is worthless if it does not leave the RIGHT spelling working."""
        assert self._make(seed=42).seed == 42

    def test_cross_family_kwarg_is_accepted(self) -> None:
        """Pins the polarity: the test is the VOCABULARY, not this family's signature.

        ``_resolve_process_kwargs`` builds ONE family-agnostic dict and hands it to
        whichever family the YAML names, so ``density_power`` -- read by the
        variable-density families, not by ``random_cartesian`` -- arriving here is the
        dispatch pattern working as designed. Tightening this to a per-family signature
        check outlaws that dispatch and fails 15 of the 19 families on the cohort's own
        kwargs; it is not a stricter version of this rule but a different, wrong one.
        Knobs a family accepts and then discards are issue #1082.
        """
        assert self._make(density_power=1.6) is not None

    def test_vocabulary_is_not_snapshotted_at_import(self) -> None:
        """A family registered after import must widen the vocabulary, not be judged stale."""
        vocabulary = _accelerator_kwarg_vocabulary()
        assert {"seed", "center_fraction", "density_power"} <= vocabulary
        assert "mask_seed" not in vocabulary


class TestAccelerationScheduleDispatch:
    """``_normalized_progress`` is exhaustive over the schedule vocabulary (#789).

    The defect: the ramp handled ``power_law`` and ``exponential`` and let
    everything else fall through to ``return ratio``. That made ``polynomial`` a
    byte-identical alias of ``linear`` on all 19 registered accelerators, while
    ``schedule_power`` -- documented as the *polynomial* schedule's own knob --
    was the one parameter it never read. The ladder is the training curriculum
    for every timestep-conditioned k-space diffusion arm, so a declared
    curriculum was being silently replaced by a different one.
    """

    BLOCK: ClassVar[dict] = {
        "num_timesteps": 5,
        "max_acceleration": 8.0,
        "base_acceleration": 1.0,
        "center_fraction": 0.08,
        "schedule_power": 3.0,
        "seed": 42,
    }

    def _curve(self, schedule: str) -> list[float]:
        accel = create_kspace_accelerator(
            "random_cartesian", acceleration_schedule=schedule, **self.BLOCK
        )
        inner = getattr(accel, "accelerator", accel)
        return [inner.get_acceleration_factor(t) for t in range(self.BLOCK["num_timesteps"])]

    def test_polynomial_is_no_longer_linear(self) -> None:
        """The regression itself: these two were identical at every rung."""
        assert self._curve("polynomial") != self._curve("linear")

    def test_polynomial_matches_power_law(self) -> None:
        """Both are ``ratio ** schedule_power``.

        This is the reading ``schedule_power``'s own description implies and the
        one ``severity.shape_severity_ratio`` has always computed; #789 settled
        the disagreement in its favour.
        """
        assert self._curve("polynomial") == pytest.approx(self._curve("power_law"))

    def test_polynomial_reads_schedule_power(self) -> None:
        """The knob must actually move the curve -- otherwise it is still unread."""
        curves = []
        for power in (2.0, 4.0):
            accel = create_kspace_accelerator(
                "random_cartesian",
                acceleration_schedule="polynomial",
                **{**self.BLOCK, "schedule_power": power},
            )
            inner = getattr(accel, "accelerator", accel)
            curves.append([inner.get_acceleration_factor(t) for t in range(5)])
        assert curves[0] != curves[1]

    def test_linear_is_unchanged(self) -> None:
        """Polarity guard: the fix must not perturb the schedule the corpus uses.

        ``linear`` is an explicit branch now rather than the fall-through, and
        that must be a no-op -- an affine ramp from base to max.
        """
        assert self._curve("linear") == pytest.approx([1.0, 2.75, 4.5, 6.25, 8.0])

    def test_step_still_receives_the_unshaped_ratio(self) -> None:
        """``step`` also returned the raw ratio before, and must keep doing so.

        Its consumer treats the value as a ladder INDEX
        (``int(ratio * len(range))``), not an interpolation weight. Shaping it
        would silently move which rung each timestep lands on -- so making the
        dispatch exhaustive had to preserve this case, not "fix" it.
        """
        accel = create_kspace_accelerator(
            "random_cartesian",
            acceleration_schedule="step",
            acceleration_range=[2.0, 8.0, 32.0],
            **self.BLOCK,
        )
        inner = getattr(accel, "accelerator", accel)
        # ratio = t/4 = [0, .25, .5, .75, 1]; idx = min(int(ratio * 3), 2).
        # Any shaping of `ratio` would move these boundaries.
        rungs = [inner.get_acceleration_factor(t) for t in range(5)]
        assert rungs == [2.0, 2.0, 8.0, 32.0, 32.0]

    @pytest.mark.parametrize("schedule", ["linear", "polynomial", "power_law", "exponential"])
    def test_forward_and_inverse_agree(self, schedule: str) -> None:
        """``timestep_for_acceleration`` had the SAME fall-through as the ramp.

        Fixing only the forward direction would leave ``polynomial`` ramping as a
        power and inverting as a line -- exactly the desynchronisation that
        method's docstring says it exists to prevent.
        """
        accel = create_kspace_accelerator(
            "random_cartesian", acceleration_schedule=schedule, **self.BLOCK
        )
        inner = getattr(accel, "accelerator", accel)
        for t in range(self.BLOCK["num_timesteps"]):
            assert inner.timestep_for_acceleration(inner.get_acceleration_factor(t)) == t

    @pytest.mark.parametrize(
        "schedule", ["linear", "polynomial", "power_law", "exponential", "step"]
    )
    def test_every_schedule_runs_from_base_at_zero_to_max_at_last_timestep(
        self, schedule: str
    ) -> None:
        """The degradation ladder is indexed from t=0, so R(0) is the base and
        R(T-1) -- not R(T) -- is the maximum, with no dip in between. Measured
        at the reverse cohort's shape (T=29, 1x..32x)."""
        extra = {"acceleration_range": [1.0, 2.0, 8.0, 32.0]} if schedule == "step" else {}
        accel = create_kspace_accelerator(
            "random_cartesian",
            acceleration_schedule=schedule,
            num_timesteps=29,
            max_acceleration=32.0,
            base_acceleration=1.0,
            center_fraction=0.08,
            schedule_power=2.0,
            seed=42,
            **extra,
        )
        inner = getattr(accel, "accelerator", accel)
        curve = [inner.get_acceleration_factor(t) for t in range(29)]
        assert curve[0] == pytest.approx(1.0)
        assert curve[-1] == pytest.approx(32.0)
        assert all(b >= a for a, b in itertools.pairwise(curve)), curve
        # A timestep past the end clamps onto the last rung rather than overshooting.
        assert inner.get_acceleration_factor(29) == pytest.approx(32.0)

    COHORT_RUNGS: ClassVar[list[float]] = [
        1.0, 2.0, 2.226, 2.485, 2.753, 3.048, 3.413, 3.765, 4.197, 4.655,
        5.224, 5.818, 6.4, 7.111, 8.0, 8.828, 9.846, 10.667, 11.636, 12.8,
        14.222, 16.0, 18.286, 19.692, 21.333, 23.273, 25.6, 28.444, 32.0,
    ]  # fmt: skip
    #: R_k = 256 / round(256 * 2^(-k/6)): geometric in line count end to end.
    EVEN_RUNGS: ClassVar[list[float]] = [
        1.0, 1.123, 1.261, 1.414, 1.59, 1.778, 2.0, 2.246, 2.51, 2.813, 3.16,
        3.556, 4.0, 4.491, 5.02, 5.689, 6.4, 7.111, 8.0, 8.828, 10.24, 11.13,
        12.8, 14.222, 16.0, 18.286, 19.692, 23.273, 25.6, 28.444, 32.0,
    ]  # fmt: skip

    @staticmethod
    def _process(rungs: list[float]):
        from spectramr.models.diffusion.kspace_process import KSpaceUndersamplingProcess

        return KSpaceUndersamplingProcess(
            num_timesteps=len(rungs),
            max_acceleration=32.0,
            base_acceleration=1.0,
            center_fraction=0.08,
            min_center_fraction=0.02,
            mask_type="density_nested",
            schedule_type="step",
            schedule_kwargs={"acceleration_range": rungs},
            enforce_nested=True,
            mask_direction="phase",
            phase_encode_axis=-1,
        )

    @pytest.mark.parametrize("ladder", ["COHORT_RUNGS", "EVEN_RUNGS"])
    def test_the_reverse_step_ladders_realise_distinct_nested_masks(self, ladder: str) -> None:
        """Every step of the reverse arms' ladders reveals new lines, and the
        fully sampled mask sits at t=0."""
        rungs = getattr(self, ladder)
        process = self._process(rungs)
        T = len(rungs)
        masks = [process.mask_at(t, (256, 256)).reshape(256, 256).bool() for t in range(T)]
        assert bool(masks[0].all()), "t=0 must be the fully sampled mask"
        for t in range(T - 1):
            assert not bool((masks[t + 1] & ~masks[t]).any()), f"t={t + 1} not nested in t={t}"
            assert not torch.equal(masks[t + 1], masks[t]), f"t={t}->{t + 1} is inert"

    @pytest.mark.parametrize(
        ("ladder", "anchors"),
        [("COHORT_RUNGS", {2.0: 1, 8.0: 14, 32.0: 28}), ("EVEN_RUNGS", {2.0: 6, 8.0: 18, 32.0: 30})],
    )
    def test_the_validation_levels_sit_on_exact_rungs(
        self, ladder: str, anchors: dict[float, int]
    ) -> None:
        """The validation cascade skips a level that inverts to no exact rung,
        so 2/8/32 must each own one."""
        process = self._process(getattr(self, ladder))
        acc = process.mask_generator._get_accelerator(process.mask_type)
        assert {r: acc.timestep_for_acceleration(r) for r in anchors} == anchors

    def test_the_even_ladder_arm_declares_the_pinned_rungs(self) -> None:
        import yaml

        from tests.utils.corpus import repo_root

        arm = repo_root() / (
            "experiments/inprogress/kspace_filling/ablations/reverse/"
            "experiment_11_reverse_additive_normalized_even_ladder.yaml"
        )
        if not arm.is_file():
            pytest.skip(f"corpus not present: {arm}")
        cfg = yaml.safe_load(arm.read_text())
        assert cfg["undersampling"]["acceleration_range"] == self.EVEN_RUNGS
        assert cfg["training"]["diffusion"]["timesteps"] == len(self.EVEN_RUNGS)

    def test_unknown_schedule_raises(self) -> None:
        """Non-negotiable #3: an unknown enum value raises, never degrades."""
        with pytest.raises(ValueError, match="unknown acceleration_schedule"):
            create_kspace_accelerator(
                "random_cartesian", acceleration_schedule="cosine", **self.BLOCK
            )

    def test_enum_member_normalises_to_its_value(self) -> None:
        """``AccelerationSchedule`` is ``(str, Enum)``, so ``str()`` is a trap.

        ``str(AccelerationSchedule.STEP)`` is ``'AccelerationSchedule.STEP'``
        while ``== 'step'`` is True. ``models/diffusion/kspace_process.py``
        compares through ``str()``, so an un-normalised enum silently failed that
        check. Normalising once at construction closes it.
        """
        from spectramr.config.schemas.enums import AccelerationSchedule

        accel = create_kspace_accelerator(
            "random_cartesian", acceleration_schedule=AccelerationSchedule.STEP, **self.BLOCK
        )
        inner = getattr(accel, "accelerator", accel)
        assert str(inner.acceleration_schedule) == "step"


class TestMinCenterFractionIsRead:
    """``nested`` / ``linear`` / ``low_pass`` honour the ACS floor (#1159).

    All three capped their realised acceleration at ``1/center_fraction`` no
    matter what the ladder declared, because they painted the ACS at the full
    ``center_fraction`` at every rung. ``nested`` is the sharp case: the
    accelerator *named* for the cold-diffusion nesting property could not reach
    the acceleration its own arm declared.
    """

    PATTERNS: ClassVar[tuple[str, ...]] = ("nested", "linear", "low_pass")

    def _top_realised(self, pattern: str, min_cf: float | None, cf: float = 0.08) -> float:
        kwargs = {
            "num_timesteps": 28,
            "max_acceleration": 32.0,
            "base_acceleration": 2.0,
            "center_fraction": cf,
            "acceleration_schedule": "step",
            "acceleration_range": [2.0, 8.0, 32.0],
            "seed": 42,
        }
        if min_cf is not None:
            kwargs["min_center_fraction"] = min_cf
        accel = create_kspace_accelerator(pattern, **kwargs)
        inner = getattr(accel, "accelerator", accel)
        best = 0.0
        for t in range(28):
            mask = inner.get_acceleration_mask(SHAPE, t, torch.device("cpu"))
            kept = max(int(mask[0].float().sum()), 1)
            best = max(best, mask[0].numel() / kept)
        return best

    @pytest.mark.parametrize("pattern", PATTERNS)
    def test_floor_lifts_the_ceiling(self, pattern: str) -> None:
        """A floor below ``center_fraction`` must let the top rung be reached."""
        assert self._top_realised(pattern, 0.02) >= 32.0 * 0.9

    @pytest.mark.parametrize("pattern", PATTERNS)
    def test_unreachable_floor_raises(self, pattern: str) -> None:
        """A floor of 0.08 cannot fit the 1/32 budget, so it is refused at build (#2516).

        These families accepted it and then capped every higher rung at
        ``1/center_fraction``, reporting a 32x ladder that realised 12.6x.
        """
        with pytest.raises(ValueError, match="exceeds the sampling budget"):
            self._top_realised(pattern, 0.08)

    @pytest.mark.parametrize("pattern", PATTERNS)
    def test_undeclared_floor_keeps_the_center_fraction_cap(self, pattern: str) -> None:
        """No declared floor: the core stays at ``center_fraction``, as the corpus expects."""
        assert self._top_realised(pattern, None) < 16.0

    @pytest.mark.parametrize("pattern", PATTERNS)
    def test_nesting_survives_the_shrinking_acs(self, pattern: str) -> None:
        """The risk this fix had to avoid: trading a ceiling for a leak.

        ``linear`` ranks the ACS *complement* once and is only nested while that
        complement stays fixed, which is why all three take the constant
        ``_guaranteed_core_fraction`` rather than a per-timestep shrink.
        """
        accel = create_kspace_accelerator(
            pattern,
            num_timesteps=28,
            max_acceleration=32.0,
            base_acceleration=2.0,
            center_fraction=0.08,
            min_center_fraction=0.02,
            acceleration_schedule="step",
            acceleration_range=[2.0, 8.0, 32.0],
            seed=42,
        )
        inner = getattr(accel, "accelerator", accel)
        masks = [
            inner.get_acceleration_mask(SHAPE, t, torch.device("cpu"))[0] > 0 for t in range(28)
        ]
        union = masks[0].clone()
        for t in range(1, 28):
            re_added = int((masks[t] & ~union).sum())
            assert re_added == 0, f"{pattern}: {re_added} bins re-entered the mask at t={t}"
            union &= masks[t]


class TestPartialFourierReportsDeclaredAcceleration:
    """``partial_fourier`` no longer overwrites ``R_nominal`` (#1160).

    It returned ``1 / pf_fraction`` from ``get_acceleration_factor``, which reads
    as honesty -- it is what the mask realises -- but is what made the pattern
    invisible to its own gate. ``describe_ladder`` sources ``R_nominal`` there,
    so nominal and effective were equal at every rung, ``declared_ladder_defects``
    saw zero drift, and ``check_acceleration_ladder_realisable.py`` passed an arm
    whose entire ladder had been discarded (pitfall #16).
    """

    def _accel(self, **overrides):
        kwargs = {
            "num_timesteps": 28,
            "max_acceleration": 32.0,
            # Below the 2.0 partial-Fourier bound that __init__ clamps
            # `max_acceleration` to, so the nominal ramp has a non-zero span to
            # traverse. That clamp is a separate silent discard, deliberately
            # left alone: `test_acceleration_levels.TestPartialFourier` pins the
            # "fixed ~75% regardless of target" behaviour as intended.
            "base_acceleration": 1.0,
            "center_fraction": 0.08,
            "acceleration_schedule": "linear",
            "seed": 42,
        }
        kwargs.update(overrides)
        accel = create_kspace_accelerator("partial_fourier", **kwargs)
        return getattr(accel, "accelerator", accel)

    def test_nominal_tracks_the_declared_ladder(self) -> None:
        inner = self._accel()
        assert inner.get_acceleration_factor(0) != inner.get_acceleration_factor(27)

    def test_declared_ladder_defects_can_now_see_it(self) -> None:
        """The point of the change: the gate's own check must stop returning clean.

        ``declared_ladder_defects`` is scoped to ``step`` with an explicit range,
        which is where the step branch reads the rungs directly and the clamp
        does not flatten them. It returned ``[]`` before this fix.
        """
        from spectramr.config.schemas.acceleration import AccelerationConfigSchema
        from spectramr.models.diffusion.kspace_process import (
            KSpaceUndersamplingProcess,
            resolve_undersampling_kwargs,
        )

        block = {
            "max_acceleration": 32.0,
            "base_acceleration": 2.0,
            "center_fraction": 0.08,
            "acceleration_range": [2.0, 8.0, 32.0],
            "seed": 42,
            "acceleration_type": "partial_fourier",
            "schedule_type": "step",
        }
        resolved = resolve_undersampling_kwargs(AccelerationConfigSchema(**block))
        process = KSpaceUndersamplingProcess(num_timesteps=28, **resolved)
        assert process.declared_ladder_defects((256, 256))

    def test_realised_acceleration_is_still_available(self) -> None:
        """The constant it truly delivers stays reachable, just not as R_nominal."""
        assert self._accel().realised_acceleration == pytest.approx(1.0 / 0.75)

    def test_mask_is_unaffected(self) -> None:
        """The fix reports, it does not re-sample: the block is still fixed.

        ``get_acceleration_mask`` reads ``pf_fraction`` directly and never routed
        through ``get_acceleration_factor``, which is precisely why this half of
        #1160 is safe to land on its own.
        """
        inner = self._accel()
        counts = {
            int(inner.get_acceleration_mask(SHAPE, t, torch.device("cpu"))[0].float().sum())
            for t in range(0, 28, 7)
        }
        assert len(counts) == 1


# ---------------------------------------------------------------------------
# Device-independent ranking (#1510)
#
# ``torch.Generator(device=...)`` is a device-specific stream, so the Gumbel
# top-K draw in ``_ranking`` produced a DIFFERENT weighted permutation on CUDA
# than on CPU for one declared ``mask_seed``. Measured on the exp_11 ladder
# (T=29, 256x256, seed 42): 28 of 29 timesteps realised different masks,
# 383,488 bins apart -- yet with IDENTICAL cardinality at every rung and
# nesting intact on both, which is precisely why it read as reproducible.
#
# The whole ranking is now built on CPU and only the finished index tensor is
# moved: drawing on CPU alone would still leave ``argsort`` tie-breaking, which
# is not guaranteed identical across devices.
#
# Characterised before changing, on this host's RTX 4080 (sha256 over the
# stacked cascade): CPU f475dbe5... before AND after -- unchanged, which is what
# sim2rank's CPU-canonical backend depends on. CUDA 65b6eb32... -> f475dbe5...,
# so GPU runs now reproduce the CPU realisation instead of drawing a second,
# undeclared one. GPU realisations of already-run density_nested arms therefore
# change; CPU ones do not.
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cross-device parity needs CUDA")
@pytest.mark.parametrize("line_axis", [None, "y"])
def test_ranking_is_bit_identical_across_devices(line_axis):
    """One seed, one permutation -- whichever device asked for it."""
    accel = _density_nested(line_axis=line_axis)
    cpu_ranking, cpu_core = accel._ranking(64, 64, torch.device("cpu"))
    cuda_ranking, cuda_core = accel._ranking(64, 64, torch.device("cuda"))

    assert cuda_ranking.device.type == "cuda"
    assert cpu_core == cuda_core
    assert torch.equal(cpu_ranking, cuda_ranking.cpu())


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cross-device parity needs CUDA")
def test_realised_masks_match_across_devices():
    """The property that actually matters to a run: the MASK, not the ranking."""
    accel = _density_nested()
    for timestep in (0, 7, 14, 27):
        on_cpu = accel.get_acceleration_mask((1, 64, 64), timestep, torch.device("cpu"))
        on_cuda = accel.get_acceleration_mask((1, 64, 64), timestep, torch.device("cuda"))
        assert torch.equal(on_cpu.bool(), on_cuda.bool().cpu()), f"diverged at t={timestep}"


def test_ranking_lands_on_the_requested_device():
    """The move is the last step, not a forgotten one."""
    accel = _density_nested()
    ranking, _ = accel._ranking(32, 32, torch.device("cpu"))
    assert ranking.device.type == "cpu"


def test_different_seeds_still_give_different_rankings():
    """Discrimination: parity must not have been bought with a constant draw.

    Without this, replacing the Gumbel draw with ``torch.arange`` would satisfy
    every parity assertion above.
    """
    first, _ = _density_nested(seed=42)._ranking(32, 32, torch.device("cpu"))
    second, _ = _density_nested(seed=1234)._ranking(32, 32, torch.device("cpu"))
    assert not torch.equal(first, second)


def test_same_seed_gives_the_same_ranking_on_two_instances():
    """The reproducibility promise itself, on the device sim2rank runs on."""
    first, _ = _density_nested(seed=42)._ranking(32, 32, torch.device("cpu"))
    second, _ = _density_nested(seed=42)._ranking(32, 32, torch.device("cpu"))
    assert torch.equal(first, second)


# ---------------------------------------------------------------------------
# Seed-keyed memo growth (#1524)
#
# ``_ranking`` MUST key on ``seed`` -- ``_generate_batch_masks_dynamic``
# (``models/diffusion/kspace_process.py:1172``) mutates ``.seed`` per sample, and
# a shape-only key would make ``enable_dynamic_mask`` a silent no-op. That
# correctness requirement is what made the memo unbounded: one permanent,
# never-reread entry per sample for the life of the run. Measured at 256x256
# with ``line_axis=None``: 524288 B/entry, so 30000 iterations at batch 2 grew
# the dict by ~29 GiB. Both properties are pinned here, because fixing either
# one alone reintroduces the other.
# ---------------------------------------------------------------------------


def test_ranking_cache_is_bounded_under_per_sample_seed_mutation():
    """The leak itself: distinct seeds must not accrue entries without limit."""
    from spectramr.infrastructure.physics.bounded_cache import DEFAULT_CACHE_CAPACITY

    accel = _density_nested()
    device = torch.device("cpu")
    for seed in range(DEFAULT_CACHE_CAPACITY * 10):
        accel.seed = seed
        accel._ranking(64, 64, device)
    assert len(accel._ranking_cache) <= DEFAULT_CACHE_CAPACITY


def test_ranking_cache_rejects_a_plain_dict_substitution():
    """A plain ``dict`` passes every behavioural assertion above until it OOMs.

    So the bound is asserted structurally too: the attribute must be a type that
    *cannot* grow without limit, not merely one that happens not to have yet.
    """
    from spectramr.infrastructure.physics.bounded_cache import BoundedLRUCache

    accel = _density_nested()
    assert isinstance(accel._ranking_cache, BoundedLRUCache)
    assert not isinstance(accel._ranking_cache, dict)


def test_fixed_seed_cascade_still_hits_the_cache():
    """Bounding must not cost the reuse the memo exists for.

    Observed via the draw itself rather than via ``len(cache)``: a cache that
    silently recomputed on every call would still report size 1.
    """
    accel = _density_nested()
    device = torch.device("cpu")
    calls = []
    original = accel._make_generator
    accel._make_generator = lambda *a, **k: (calls.append(1), original(*a, **k))[1]

    for _ in range(accel.num_timesteps):
        accel._ranking(64, 64, device)

    assert len(calls) == 1, f"fixed-seed cascade recomputed {len(calls)} times"


def test_ranking_cache_still_varies_with_seed_after_bounding():
    """The guard the bound must not break: seed stays in the key."""
    accel = _density_nested()
    device = torch.device("cpu")
    accel.seed = 7
    first, _ = accel._ranking(32, 32, device)
    accel.seed = 8
    second, _ = accel._ranking(32, 32, device)
    assert not torch.equal(first, second)


def test_cold_diffusion_nested_cache_is_bounded():
    """The sibling memo, keyed on ``seed`` the same way.

    ``enforce_nested`` is switched off on the seed-mutating path today, so this
    one does not grow in production -- it is bounded so that re-enabling
    enforcement cannot reintroduce the leak without anything going red.
    """
    from spectramr.infrastructure.physics.bounded_cache import (
        DEFAULT_CACHE_CAPACITY,
        BoundedLRUCache,
    )
    from spectramr.infrastructure.physics.sampling import ColdDiffusionAccelerator

    accel = ColdDiffusionAccelerator(
        num_timesteps=4, max_acceleration=8.0, acceleration_type="density_nested"
    )
    assert isinstance(accel._nested_cache, BoundedLRUCache)
    assert accel._nested_cache.capacity == DEFAULT_CACHE_CAPACITY


# ---------------------------------------------------------------------------
# VariableDensityKSpaceAccelerator: seed-keyed ranking and the ACS floor
# (#1071, #1070, #1257); NestedKSpaceAccelerator's seed-keyed permutation (#1071)
# ---------------------------------------------------------------------------

_LADDER = [float(r) for r in range(2, 33)]


def _vd(**overrides):
    from spectramr.infrastructure.physics.sampling import VariableDensityKSpaceAccelerator

    kwargs = {
        "num_timesteps": 28,
        "max_acceleration": 32.0,
        "center_fraction": 0.08,
        "seed": 42,
    }
    kwargs.update(overrides)
    return VariableDensityKSpaceAccelerator(**kwargs)


def _vd_ladder(**overrides):
    """One rung per timestep over every integer R in 2..32."""
    kwargs = {
        "num_timesteps": len(_LADDER),
        "base_acceleration": 2.0,
        "acceleration_range": _LADDER,
        "acceleration_schedule": "step",
        "min_center_fraction": 0.03,
    }
    kwargs.update(overrides)
    return _vd(**kwargs)


def _core_disc(fraction: float, height: int = 256, width: int = 256) -> torch.Tensor:
    """The equal-area centred disc the floor declares (the #581 ACS geometry)."""
    radius = math.sqrt(max(height * width * fraction, 1.0) / math.pi)
    rows = torch.arange(height, dtype=torch.float32) - height // 2
    cols = torch.arange(width, dtype=torch.float32) - width // 2
    return torch.sqrt(rows[:, None] ** 2 + cols[None, :] ** 2) <= radius


def _floor_met_at_every_rung(accel, fraction: float) -> bool:
    core = _core_disc(fraction)
    for t in range(accel.num_timesteps):
        mask = accel.get_acceleration_mask(SHAPE, t)[0]
        if not bool(mask[core].all()):
            return False
    return True


def _legacy_seedless_key(self, channels, height, width, device):
    """The pre-#1071 memo key, which omitted the seed."""
    return (channels, height, width, str(device))


class TestVariableDensitySeed:
    def test_two_seeds_give_different_rankings_and_masks(self) -> None:
        a, b = _vd(seed=42), _vd(seed=999)
        cpu = torch.device("cpu")
        assert not torch.equal(
            a._priority_ranking(1, 256, 256, cpu), b._priority_ranking(1, 256, 256, cpu)
        )
        assert not torch.equal(
            a.get_acceleration_mask(SHAPE, 20), b.get_acceleration_mask(SHAPE, 20)
        )

    def test_same_seed_is_reproducible_across_instances(self) -> None:
        assert torch.equal(
            _vd(seed=7).get_acceleration_mask(SHAPE, 20),
            _vd(seed=7).get_acceleration_mask(SHAPE, 20),
        )

    def test_mutating_seed_redraws_and_restoring_it_reproduces(self) -> None:
        """The dynamic-mask path mutates ``.seed`` on one instance per sample."""
        accel = _vd(seed=42)
        first = accel.get_acceleration_mask(SHAPE, 20).clone()
        accel.seed = 999
        assert not torch.equal(first, accel.get_acceleration_mask(SHAPE, 20))
        accel.seed = 42
        assert torch.equal(first, accel.get_acceleration_mask(SHAPE, 20))

    def test_planted_seedless_key_fails_the_redraw_property(self, monkeypatch) -> None:
        """A memo key without the seed replays the first draw for every sample."""
        from spectramr.infrastructure.physics.sampling import VariableDensityKSpaceAccelerator

        monkeypatch.setattr(
            VariableDensityKSpaceAccelerator, "_priority_cache_key", _legacy_seedless_key
        )
        accel = _vd(seed=42)
        first = accel.get_acceleration_mask(SHAPE, 20).clone()
        accel.seed = 999
        assert torch.equal(first, accel.get_acceleration_mask(SHAPE, 20))

    def test_cache_is_bounded(self) -> None:
        from spectramr.infrastructure.physics.bounded_cache import BoundedLRUCache

        assert isinstance(_vd()._priority_cache, BoundedLRUCache)


class TestVariableDensityAcsFloor:
    @pytest.mark.parametrize("density_power", [0.0, 2.0])
    def test_floor_is_fully_sampled_at_every_rung_r2_to_r32(self, density_power: float) -> None:
        """A flat density (power 0) leaves the core prefix as the only thing holding the floor."""
        accel = _vd_ladder(density_power=density_power)
        total = SHAPE[1] * SHAPE[2]
        realised = [
            total / int(accel.get_acceleration_mask(SHAPE, t).sum())
            for t in range(accel.num_timesteps)
        ]
        assert realised[0] == pytest.approx(2.0, rel=1e-3)
        assert realised[-1] == pytest.approx(32.0, rel=1e-3)
        assert _floor_met_at_every_rung(accel, 0.03)

    def test_planted_coreless_ranking_misses_the_floor(self, monkeypatch) -> None:
        """Without the core prefix a flat density hole-punches the declared ACS."""
        from spectramr.infrastructure.physics.sampling import VariableDensityKSpaceAccelerator

        monkeypatch.setattr(
            VariableDensityKSpaceAccelerator, "_guaranteed_core_fraction", lambda self, cf: 0.0
        )
        assert not _floor_met_at_every_rung(_vd_ladder(density_power=0.0), 0.03)

    def test_center_fraction_is_the_core_when_no_floor_is_declared(self) -> None:
        accel = _vd(center_fraction=0.05, density_power=0.0, max_acceleration=8.0, num_timesteps=4)
        assert bool(accel.get_acceleration_mask(SHAPE, 3)[0][_core_disc(0.05)].all())

    @pytest.mark.parametrize(
        "cls_name", ["VariableDensityKSpaceAccelerator", "FractionalVariableDensityAccelerator"]
    )
    def test_impossible_floor_raises(self, cls_name: str) -> None:
        from spectramr.infrastructure.physics import sampling

        cls = getattr(sampling, cls_name)
        with pytest.raises(ValueError, match="exceeds the sampling budget"):
            cls(
                num_timesteps=28,
                max_acceleration=32.0,
                center_fraction=0.08,
                min_center_fraction=0.08,
                seed=1,
            )

    def test_floor_at_the_budget_is_accepted(self) -> None:
        assert _vd(min_center_fraction=1.0 / 32.0).min_center_fraction == pytest.approx(1 / 32)


def test_nested_permutation_redraws_per_seed() -> None:
    from spectramr.infrastructure.physics.bounded_cache import BoundedLRUCache
    from spectramr.infrastructure.physics.sampling import NestedKSpaceAccelerator

    accel = NestedKSpaceAccelerator(
        num_timesteps=28, max_acceleration=32.0, center_fraction=0.02, seed=42
    )
    assert isinstance(accel._permutation_cache, BoundedLRUCache)
    first = accel.get_acceleration_mask(SHAPE, 20).clone()
    accel.seed = 999
    assert not torch.equal(first, accel.get_acceleration_mask(SHAPE, 20))
    accel.seed = 42
    assert torch.equal(first, accel.get_acceleration_mask(SHAPE, 20))


_NESTED_FAMILIES = ["NestedKSpaceAccelerator", "DensityNestedKSpaceAccelerator"]


def _nested_family(cls_name: str, **overrides):
    from spectramr.infrastructure.physics import sampling

    kwargs = {"num_timesteps": 28, "max_acceleration": 32.0, "center_fraction": 0.08, "seed": 1}
    kwargs.update(overrides)
    return getattr(sampling, cls_name)(**kwargs)


class TestNestedFamiliesAcsFloor:
    """#2516: both fixed-ranking nested families refuse a floor the top budget cannot hold."""

    @pytest.mark.parametrize("cls_name", _NESTED_FAMILIES)
    def test_impossible_floor_raises(self, cls_name: str) -> None:
        with pytest.raises(ValueError, match="exceeds the sampling budget"):
            _nested_family(cls_name, min_center_fraction=0.08)

    @pytest.mark.parametrize("cls_name", _NESTED_FAMILIES)
    def test_floor_at_the_budget_is_accepted(self, cls_name: str) -> None:
        accel = _nested_family(cls_name, min_center_fraction=1.0 / 32.0)
        assert accel.min_center_fraction == pytest.approx(1.0 / 32.0)

    @pytest.mark.parametrize("cls_name", _NESTED_FAMILIES)
    def test_undeclared_floor_never_raises(self, cls_name: str) -> None:
        """The implicit core is ``center_fraction`` and is truncated by budget, not rejected."""
        assert _nested_family(cls_name).min_center_fraction is None

    @pytest.mark.parametrize(
        ("cls_name", "realised_top_r"),
        [("NestedKSpaceAccelerator", 12.64), ("DensityNestedKSpaceAccelerator", 32.0)],
    )
    def test_planted_unvalidated_floor_is_silently_mishandled(
        self, monkeypatch, cls_name: str, realised_top_r: float
    ) -> None:
        """Without the build-time check the declaration is accepted and then broken.

        ``nested`` lets the ACS override the requested acceleration; ``density_nested``
        truncates the core, so the declared floor is not delivered. Either way the run
        reports the declared ladder while realising something else.
        """
        from spectramr.infrastructure.physics.sampling import KSpaceAccelerator

        monkeypatch.setattr(KSpaceAccelerator, "_validate_center_floor", lambda self, r: None)
        accel = _nested_family(cls_name, min_center_fraction=0.08)
        mask = accel.get_acceleration_mask(SHAPE, accel.num_timesteps - 1)
        total = SHAPE[1] * SHAPE[2]
        assert total / int(mask.sum()) == pytest.approx(realised_top_r, rel=1e-2)
        if cls_name == "DensityNestedKSpaceAccelerator":
            assert not bool(mask[0][_core_disc(0.08)].all())


def _registered_families() -> list[str]:
    from spectramr.infrastructure.physics.sampling import _ACCELERATOR_REGISTRY

    return sorted(_ACCELERATOR_REGISTRY)


#: Caps its own top acceleration at 2x, so a 0.04 floor is inside its budget.
_FLOOR_FITS_BY_GEOMETRY = {"partial_fourier"}


@pytest.mark.parametrize("pattern", _registered_families())
def test_every_family_answers_one_floor_declaration_the_same_way(pattern: str) -> None:
    """#2061: one declaration, one answer, whichever family the arm picked."""
    kwargs = {
        "num_timesteps": 28,
        "max_acceleration": 32.0,
        "base_acceleration": 2.0,
        "center_fraction": 0.04,
        "seed": 1,
    }
    create_kspace_accelerator(pattern, **kwargs)
    if pattern in _FLOOR_FITS_BY_GEOMETRY:
        create_kspace_accelerator(pattern, min_center_fraction=0.04, **kwargs)
        return
    with pytest.raises(ValueError, match="exceeds the sampling budget"):
        create_kspace_accelerator(pattern, min_center_fraction=0.04, **kwargs)


def test_planted_guard_off_the_base_lets_families_diverge(monkeypatch) -> None:
    """With the base no longer checking, the families that relied on it accept the floor."""
    from spectramr.infrastructure.physics.sampling import KSpaceAccelerator

    monkeypatch.setattr(KSpaceAccelerator, "_validate_center_floor", lambda self, r: None)
    for pattern in ("uniform_cartesian", "poisson_disk", "variable_density_2d_gaussian"):
        create_kspace_accelerator(
            pattern,
            num_timesteps=28,
            max_acceleration=32.0,
            base_acceleration=2.0,
            center_fraction=0.04,
            min_center_fraction=0.04,
            seed=1,
        )


def test_family_reads_line_axis_matches_what_each_family_builds() -> None:
    """#1082: the signature walk agrees with the built accelerator for every family.

    A family that takes ``line_axis`` exposes it after construction; one that drops
    it in ``**kwargs`` does not. ``multi_mask`` refuses a direction outright.
    """
    from spectramr.infrastructure.physics import sampling as S

    for name in sorted(S._ACCELERATOR_REGISTRY):
        claimed = S.family_reads_line_axis(name)
        try:
            built = S.ColdDiffusionAccelerator(
                acceleration_type=name, num_timesteps=4, max_acceleration=8.0, mask_direction="y"
            )
        except ValueError:
            assert name == "multi_mask" and not claimed
            continue
        assert claimed == (getattr(built.accelerator, "line_axis", None) is not None), name


def test_family_reads_line_axis_resolves_aliases() -> None:
    from spectramr.infrastructure.physics.sampling import family_reads_line_axis

    assert family_reads_line_axis("density_nested")
    assert not family_reads_line_axis("random")


def _spoke_mask(theta: float, size: int = 256) -> torch.Tensor:
    from spectramr.infrastructure.physics.sampling import MaskGenerator

    radius = torch.linspace(-math.pi, math.pi, 2 * size - 1)
    angle = torch.tensor(theta, dtype=torch.float32)
    trajectory = torch.stack([radius * torch.cos(angle), radius * torch.sin(angle)])
    rows, cols = MaskGenerator.trajectory_bin_indices(trajectory, size, size)
    mask = torch.zeros(size, size, dtype=torch.bool)
    mask[rows, cols] = True
    return mask


def test_rasteriser_puts_k0_on_the_centre_bin() -> None:
    from spectramr.infrastructure.physics.sampling import MaskGenerator

    rows, cols = MaskGenerator.trajectory_bin_indices(torch.zeros(2, 1), 256, 256)
    assert (rows.item(), cols.item()) == (128, 128)


def test_rasteriser_bins_a_0_and_90_degree_spoke_as_a_transpose_pair() -> None:
    """#2551: ``cos(pi/2) = -4.4e-8`` must not move a sample off the centre line.

    Planted: flooring (the pre-fix rule) and ``round(x - 0.5)`` (the rule the
    issue proposed) both leave the pair differing on 82 bins at 256x256.
    """
    assert torch.equal(_spoke_mask(0.0), _spoke_mask(math.pi / 2).T)


class TestVariableDensityPoissonDisc:
    """The static Poisson-disc family the rep-masking transform draws from."""

    @staticmethod
    def _allowed() -> torch.Tensor:
        allowed = torch.zeros(64, 64, dtype=torch.bool)
        allowed[:, 8:56] = True
        return allowed

    def _draw(self, radius: float, seed: int = 0, **kwargs) -> torch.Tensor:
        return MaskGenerator.variable_density_poisson_disc(
            self._allowed(), radius, torch.Generator().manual_seed(seed), **kwargs
        )

    def test_it_stays_inside_the_allowed_bins(self) -> None:
        assert not (self._draw(1.5) & ~self._allowed()).any()

    def test_it_is_seeded(self) -> None:
        assert torch.equal(self._draw(1.5, seed=3), self._draw(1.5, seed=3))
        assert not torch.equal(self._draw(1.5, seed=3), self._draw(1.5, seed=4))

    def test_a_larger_radius_keeps_fewer_bins(self) -> None:
        counts = [int(self._draw(r).sum()) for r in (0.5, 1.0, 1.5, 2.5)]
        assert counts == sorted(counts, reverse=True) and counts[0] > counts[-1]

    def test_no_two_kept_bins_are_closer_than_the_dc_radius(self) -> None:
        kept = torch.nonzero(self._draw(2.0, density_slope=0.0)).float()
        dist = torch.cdist(kept, kept) + torch.eye(kept.shape[0]) * 1e9
        assert float(dist.min()) >= 2.0

    def test_the_density_falls_away_from_dc(self) -> None:
        kept = self._draw(1.0).float()
        centre = kept[24:40, 24:40].mean()
        edge = torch.cat([kept[:8, 8:56].flatten(), kept[56:, 8:56].flatten()]).mean()
        assert float(centre) > 2 * float(edge)
